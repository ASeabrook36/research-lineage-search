import copy
import logging

import pytest
from azure.core.paging import ItemPaged
from azure.cosmos.exceptions import CosmosHttpResponseError, CosmosResourceNotFoundError

from research import db, search, web
from research.embed import CHUNK_VERSION, MODEL_VERSION
from research.ingest import sync_items


class Container:
    def __init__(self, name="test", rows=()):
        self.id = name
        self.rows = list(rows)
        self.queries = []
        self.reads = []
        self.upserts = []
        self.deletes = []
        self.items = {}

    def query_items(self, **kwargs):
        self.queries.append(kwargs)
        hook = kwargs["response_hook"]
        hook({"x-ms-request-charge": "99"}, ItemPaged([]))
        for start in range(0, len(self.rows), 2):
            page = self.rows[start:start + 2]
            hook({"x-ms-request-charge": "2.5"}, page)
            yield from page

    def read_item(self, item_id, *, partition_key, response_hook):
        self.reads.append((item_id, partition_key))
        if (item_id, partition_key) not in self.items:
            raise CosmosResourceNotFoundError(status_code=404)
        response_hook({"x-ms-request-charge": "1"}, self.items[item_id, partition_key])
        return self.items[item_id, partition_key]

    def upsert_item(self, item, *, response_hook):
        self.upserts.append(item)
        response_hook({"x-ms-request-charge": "5"}, item)

    def delete_item(self, item_id, *, partition_key, response_hook):
        self.deletes.append((item_id, partition_key))
        response_hook({"x-ms-request-charge": "3"}, None)


class Database:
    def __init__(self, **containers):
        self.containers = containers

    def get_container_client(self, name):
        return self.containers[name]


def complete_state(**overrides):
    return {
        "id": "ingestion-state", "status": "complete",
        "report": {
            "ingested_at": "generation-1", "model_version": MODEL_VERSION,
            "chunk_version": CHUNK_VERSION, "storage_version": db.STORAGE_VERSION,
            "papers": 150, "passages": 2500, "citation_edges": 3626, "historical_edges": 7,
            **overrides,
        },
    }


def test_query_scopes_partition_and_accumulates_every_page():
    container = Container(rows=[{"id": str(i)} for i in range(5)])
    metrics = db.RequestMetrics()
    assert len(db.query(container, "SELECT c.id FROM c", partition_key="paper", metrics=metrics)) == 5
    options = container.queries[0]
    assert options["partition_key"] == "paper"
    assert options["enable_cross_partition_query"] is False
    assert options["max_item_count"] == 100
    assert metrics.summary() == {
        "responses": 3, "request_charge": 7.5, "charge_complete": True,
        "throttle_retries": 0, "throttle_wait_ms": 0,
    }
    db.query(container, "SELECT c.id FROM c")
    assert container.queries[-1]["enable_cross_partition_query"] is True


def test_metrics_distinguish_missing_charges_from_zero():
    metrics = db.RequestMetrics()
    metrics.record({}, [])
    assert metrics.summary()["request_charge"] is None
    metrics.record({
        "x-ms-request-charge": "0", "x-ms-throttle-retry-count": "2",
        "x-ms-throttle-retry-wait-time-ms": "150",
    }, [])
    assert metrics.summary() == {
        "responses": 2, "request_charge": 0, "charge_complete": False,
        "throttle_retries": 2, "throttle_wait_ms": 150,
    }


def test_read_optional_is_a_point_read_and_only_handles_not_found():
    container = Container()
    assert db.read_optional(container, "config") is None
    assert container.reads == [("config", "config")]
    container.items["config", "config"] = {"id": "config"}
    assert db.read_optional(container, "config") == {"id": "config"}
    assert not container.queries

    class Unavailable(Container):
        def read_item(self, *args, **kwargs):
            raise CosmosHttpResponseError(status_code=429, headers={"x-ms-retry-after-ms": "100"})

    with pytest.raises(CosmosHttpResponseError):
        db.read_optional(Unavailable(), "config")


def test_incremental_sync_skips_unchanged_items_and_preserves_inputs():
    container = Container()
    items = [{"id": "same", "paper_id": "a", "text": "original"}, {"id": "same", "paper_id": "b"}]
    original = copy.deepcopy(items)
    assert sync_items(container, items, "paper_id") == {"upserted": 2, "deleted": 0, "unchanged": 0}
    assert items == original
    container.rows = copy.deepcopy(container.upserts)
    container.upserts.clear()
    assert sync_items(container, items, "paper_id") == {"upserted": 0, "deleted": 0, "unchanged": 2}
    assert not container.upserts
    items[0]["text"] = "changed"
    assert sync_items(container, items[:1], "paper_id") == {"upserted": 1, "deleted": 1, "unchanged": 0}
    assert container.deletes == [("same", "b")]
    assert container.upserts[0]["text"] == "changed"


def test_sync_migrates_unhashed_items_and_rejects_duplicates_before_writing():
    container = Container(rows=[{"id": "a"}])
    assert sync_items(container, [{"id": "a", "text": "old"}], "id")["upserted"] == 1
    with pytest.raises(ValueError, match="Duplicate"):
        sync_items(container, [{"id": "b"}, {"id": "b"}], "id")
    assert len(container.upserts) == 1


def test_public_does_not_expose_storage_metadata():
    assert search.public({"id": "x", "_etag": "etag", "sync_hash": "hash", "content_hash": "pdf"}) == {
        "id": "x", "content_hash": "pdf",
    }


def test_container_policies_cover_query_filters_without_indexing_payloads():
    paths = {entry["path"] for entry in db.INDEX_POLICY["includedPaths"]}
    assert paths == {"/*"}
    excluded = {entry["path"] for entry in db.INDEX_POLICY["excludedPaths"]}
    assert {"/embedding/*", "/text/?", "/title/?", "/sync_hash/?"} <= excluded
    assert not excluded.intersection({
        "/paper_id/?", "/model_version/?", "/year/?", "/countries/*",
        "/fields/*", "/institutions/*", "/authors/*",
    })
    assert db.INDEX_POLICY["vectorIndexes"] == [{"path": "/embedding", "type": "diskANN"}]
    assert db.CONTAINERS["edges"][0] == "from_id"
    assert db.CONTAINERS["incoming-edges"][0] == "to_id"


def test_validation_detects_existing_partition_and_index_mismatch():
    policy = db.CONTAINERS["edges"][1]

    class Definition:
        id = "edges"

        def read(self):
            return properties

    properties = {"partitionKey": {"paths": ["/from_id"]}, "indexingPolicy": copy.deepcopy(policy)}
    db.validate_container(Definition(), "from_id", policy)
    properties["partitionKey"]["paths"] = ["/kind"]
    with pytest.raises(RuntimeError, match="partition key"):
        db.validate_container(Definition(), "from_id", policy)
    properties["partitionKey"]["paths"] = ["/from_id"]
    properties["indexingPolicy"]["includedPaths"] = [{"path": "/from_id/?"}]
    with pytest.raises(RuntimeError, match="includedPaths"):
        db.validate_container(Definition(), "from_id", policy)


def test_filter_options_are_materialized_and_generation_checked(monkeypatch):
    settings = Container("settings")
    state = complete_state()
    stored_options = {"country": [{"id": "JP", "name": "JP", "count": 1}]}
    settings.items["ingestion-state", "ingestion-state"] = state
    settings.items["filter-options", "filter-options"] = {
        "ingested_at": state["report"]["ingested_at"], "options": stored_options,
    }
    monkeypatch.setattr(search, "connect", lambda: Database(settings=settings))
    assert search.options() == stored_options
    assert settings.reads == [("ingestion-state", "ingestion-state"), ("filter-options", "filter-options")]
    assert not settings.queries
    settings.items["filter-options", "filter-options"]["ingested_at"] = "old"
    with pytest.raises(RuntimeError, match="filter options"):
        search.options()


@pytest.mark.parametrize("change", [{"model_version": "old"}, {"chunk_version": "old"}, {"storage_version": 1}])
def test_incompatible_ingestion_cannot_be_searched(monkeypatch, change):
    settings = Container("settings")
    settings.items["ingestion-state", "ingestion-state"] = complete_state(**change)
    monkeypatch.setattr(search, "connect", lambda: Database(settings=settings))
    with pytest.raises(RuntimeError, match="version|schema"):
        search.require_ingestion()


def test_status_uses_published_counts_and_opt_in_live_diagnostics(monkeypatch, tmp_path):
    settings = Container("settings")
    settings.items["ingestion-state", "ingestion-state"] = complete_state()
    counts = {name: Container(name, [count]) for name, count in (("papers", 150), ("passages", 2500), ("edges", 3633))}
    monkeypatch.setattr(web, "connect", lambda: Database(settings=settings, **counts))
    monkeypatch.setattr(web, "ROOT", tmp_path)
    monkeypatch.setattr(web, "MODEL_PATH", tmp_path)
    (tmp_path / "modules.json").write_text("{}")
    response = web.status()
    assert response["ready"] is True
    assert response["counts"] == {"papers": 150, "passages": 2500, "edges": 3633}
    assert response["counts_source"] == "completed_ingestion"
    assert all(not container.queries for container in counts.values())
    assert web.status(live_counts=True)["counts_source"] == "live"
    assert all(len(container.queries) == 1 for container in counts.values())
    settings.items["ingestion-state", "ingestion-state"] = {"status": "running"}
    assert web.status()["ready"] is False


def test_vector_search_hydrates_only_selected_passages_and_logs_all_attempts(monkeypatch, caplog):
    class Candidates(Container):
        def query_items(self, **kwargs):
            if "VectorDistance" in kwargs["query"]:
                self.rows = [{"id": str(i), "paper_id": "a", "score": 1 - i / 100} for i in range(40)]
                if self.queries:
                    self.rows.append({"id": "b", "paper_id": "b", "score": 0.5})
            else:
                parameters = {p["name"]: p["value"] for p in kwargs["parameters"]}
                self.rows = [
                    self.items[value, parameters["@paper"]]
                    for name, value in parameters.items() if name.startswith("@id")
                ]
            yield from super().query_items(**kwargs)

    container = Candidates("passages")
    for item_id, paper_id in [("0", "a"), ("1", "a"), ("b", "b")]:
        container.items[item_id, paper_id] = {
            "id": item_id, "paper_id": paper_id, "text": "evidence", "title": "Paper " + paper_id,
            "year": 2000, "countries": ["JP"], "page": None, "section": "abstract",
            "source_url": "https://example.org", "coverage": "abstract", "embedding": [1, 0],
        }
    monkeypatch.setattr(search, "connect", lambda: Database(passages=container))
    monkeypatch.setattr(search, "require_ingestion", lambda: complete_state(passages=200))
    with caplog.at_level(logging.INFO, logger="research.search"):
        results, count = search.search_vector([1, 0], limit=2)
    assert [row["id"] for row in results] == ["a", "b"]
    assert count == 41
    assert not container.reads
    assert "TOP 40" in container.queries[0]["query"]
    assert "TOP 120" in container.queries[1]["query"]
    assert "c.text" in container.queries[0]["query"]
    assert "c.text" not in container.queries[1]["query"] and "c.title" not in container.queries[1]["query"]
    assert [q["partition_key"] for q in container.queries[2:]] == ["a", "b"]
    assert all("c.embedding" not in q["query"] for q in container.queries[2:])
    assert "embedding" not in results[0]["passages"][0]
    assert "attempts=2 total_candidates=81 final_candidates=41 hydrated=3" in caplog.text
    assert "'request_charge': 107.5" in caplog.text


def test_vector_first_attempt_avoids_extra_evidence_queries(monkeypatch, caplog):
    evidence = {
        "text": "evidence", "title": "Paper", "year": 2000, "countries": ["JP"],
        "page": None, "section": "abstract", "source_url": "https://example.org", "coverage": "abstract",
    }
    container = Container("passages", [
        {**evidence, "id": "a0", "paper_id": "a", "score": 1.0},
        {**evidence, "id": "a1", "paper_id": "a", "score": 0.9},
        {**evidence, "id": "b0", "paper_id": "b", "score": 0.8},
    ])
    monkeypatch.setattr(search, "connect", lambda: Database(passages=container))
    monkeypatch.setattr(search, "require_ingestion", lambda: complete_state(passages=3))
    with caplog.at_level(logging.INFO, logger="research.search"):
        results, count = search.search_vector([1, 0], limit=2)
    assert [r["id"] for r in results] == ["a", "b"]
    assert count == 3
    assert len(container.queries) == 1
    assert not container.reads
    assert "TOP 3" in container.queries[0]["query"]
    assert "hydration_queries=0" in caplog.text
