import os

import numpy as np
import pytest
from azure.cosmos.exceptions import CosmosHttpResponseError

from research.db import CONTAINERS, VECTOR_POLICY, connect, probe, query, read_item, validate_container
from research.embed import embed, model
from research.search import Filters, build_options, filter_clause, options, papers, public, search, search_vector
from research.ingest import sync_items

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(os.environ.get("RUN_INTEGRATION") != "1", reason="Set RUN_INTEGRATION=1 for populated emulator tests"),
]


def test_real_cosmos_nearest_neighbors_and_filtering():
    assert probe()["read_after_write"]


def test_population_and_source_coverage():
    records = papers()
    assert len(records) == 150
    assert sum(p["coverage"] == "full_text" for p in records) >= 20
    assert all(any(c in p["countries"] for p in records) for c in ["JP", "US", "NL"])
    assert all(1980 <= p["year"] <= 2026 for p in records)
    assert all(p["passage_count"] == 0 for p in records if p["coverage"] == "metadata")


def test_model_is_local_and_actual_passages_fit_token_limit():
    from research.collect import ROOT
    from research.embed import chunks
    import json
    tokenizer = model().tokenizer
    for paper in json.loads((ROOT / "corpus.json").read_text()):
        for passage in chunks(paper, tokenizer):
            assert len(tokenizer.encode(passage["embedding_text"])) <= 256
    assert len(embed(["A new idea"])[0]) == 384


def test_all_database_filter_categories_and_combinations():
    records = {p["id"]: p for p in papers()}
    usable = next(p for p in records.values() if p["passage_count"] and p["authors"] and p["institutions"] and p["countries"] != ["Unknown"])
    cases = [
        Filters(country=["JP"]), Filters(country=["US", "NL"]), Filters(country=["Unknown"]),
        Filters(field=[usable["fields"][0]]),
        Filters(institution=[usable["institutions"][0]["id"]]),
        Filters(author=[usable["authors"][0]["id"]]),
        Filters(year_from=1990, year_to=2000),
        Filters(country=usable["countries"], author=[usable["authors"][0]["id"]], year_from=usable["year"], year_to=usable["year"]),
    ]
    vector = embed(["EUV lithography"])[0]
    for filters in cases:
        results, _ = search_vector(vector, filters, limit=5)
        assert results
        for hit in results:
            paper = records[hit["id"]]
            for key, field in [("country", "countries"), ("field", "fields")]:
                selected = getattr(filters, key)
                assert not selected or set(selected).intersection(paper[field])
            for key, field in [("institution", "institutions"), ("author", "authors")]:
                selected = getattr(filters, key)
                assert not selected or set(selected).intersection(i["id"] for i in paper[field])
            assert filters.year_from is None or paper["year"] >= filters.year_from
            assert filters.year_to is None or paper["year"] <= filters.year_to


def test_unknown_author_and_institution_and_no_matches():
    for filters in [Filters(author=["Unknown"]), Filters(institution=["Unknown"]), Filters(country=["ZZ"])]:
        results, _ = search_vector(embed(["EUV"])[0], filters)
        if filters.country:
            assert not results
        else:
            all_papers = {p["id"]: p for p in papers()}
            field = "authors" if filters.author else "institutions"
            assert all(not all_papers[p["id"]][field] for p in results)


def test_cosmos_results_match_exact_cosine_oracle_on_small_filtered_set():
    vector = np.array(embed(["Early Japanese soft x ray imaging"])[0])
    filters = Filters(country=["JP"], year_to=2000)
    clauses, parameters = filter_clause(filters)
    rows = query(
        connect().get_container_client("passages"),
        "SELECT c.id, c.paper_id, c.embedding FROM c WHERE " + " AND ".join(clauses),
        parameters,
    )
    scores = {}
    for row in rows:
        embedding = np.array(row["embedding"])
        score = float(np.dot(vector, embedding) / (np.linalg.norm(vector) * np.linalg.norm(embedding)))
        scores[row["paper_id"]] = max(scores.get(row["paper_id"], -2), score)
    expected = sorted(scores, key=scores.get, reverse=True)[:5]
    actual, _ = search_vector(vector.tolist(), filters, limit=5)
    assert [p["id"] for p in actual] == expected
    assert all(abs(p["score"] - scores[p["id"]]) < 1e-4 for p in actual)


def test_search_is_deduplicated_and_reuses_persistent_data():
    results = search("ASML European lithography program")["results"]
    assert len({p["id"] for p in results}) == len(results)
    assert len(results) == 10
    assert all(p["passages"] for p in results)
    assert results[0]["score"] >= results[-1]["score"]
    count = query(connect().get_container_client("passages"), "SELECT VALUE COUNT(1) FROM c")[0]
    assert count >= 1000


def test_query_errors_do_not_return_success_shaped_defaults():
    with pytest.raises(CosmosHttpResponseError):
        query(connect().get_container_client("passages"), "SELECT * FROM c ORDER BY NoSuchFunction(c.id)")
    with pytest.raises(ValueError, match="Enter"):
        search(" ")


def test_idempotent_sync_removes_stale_items_in_real_cosmos():
    from azure.cosmos import PartitionKey
    db = connect()
    container = db.create_container_if_not_exists("idempotency-probe", partition_key=PartitionKey("/paper_id"))
    try:
        sync_items(container, [{"id": "old", "paper_id": "paper"}, {"id": "keep", "paper_id": "paper"}], "paper_id")
        desired = [{"id": "new", "paper_id": "paper"}, {"id": "keep", "paper_id": "paper", "updated": True}]
        assert sync_items(container, desired, "paper_id") == {"upserted": 2, "deleted": 1, "unchanged": 0}
        before = {row["id"]: row["_etag"] for row in query(container, "SELECT * FROM c")}
        assert sync_items(container, desired, "paper_id") == {"upserted": 0, "deleted": 0, "unchanged": 2}
        after = {row["id"]: row["_etag"] for row in query(container, "SELECT * FROM c")}
        assert before == after
        rows = query(container, "SELECT c.id, c.updated FROM c")
        assert {row["id"] for row in rows} == {"new", "keep"}
        assert next(row for row in rows if row["id"] == "keep")["updated"] is True
    finally:
        db.delete_container("idempotency-probe")


def test_persisted_policies_and_materialized_metadata():
    from research.web import status
    db = connect()
    for name, (partition_field, policy) in CONTAINERS.items():
        validate_container(
            db.get_container_client(name), partition_field, policy,
            VECTOR_POLICY if name == "passages" else None,
        )
    assert options() == build_options(papers())
    cached = status()
    live = status(live_counts=True)
    assert cached["ready"] and live["ready"]
    assert cached["counts"] == live["counts"]
    assert cached["counts_source"] == "completed_ingestion"
    assert live["counts_source"] == "live"


def test_partitioned_citation_views_preserve_connections_and_lineage():
    from research.web import connections, lineage
    db = connect()
    edges = query(db.get_container_client("edges"), "SELECT * FROM c")
    incoming = query(db.get_container_client("incoming-edges"), "SELECT * FROM c")
    expected_incoming = [edge for edge in edges if edge["kind"] == "citation" and edge["target_in_corpus"]]
    assert {row["id"]: public(row) for row in incoming} == {row["id"]: public(row) for row in expected_incoming}
    target = next(edge["to_id"] for edge in expected_incoming)
    response = connections(target)
    expected = [edge for edge in edges if edge["kind"] == "citation" and target in (edge["from_id"], edge["to_id"])]
    assert {edge["id"] for edge in response["citations"]} == {edge["id"] for edge in expected}
    assert all("sync_hash" not in edge for edge in response["citations"])
    graph = lineage()
    assert len(graph["citations"]) == len(expected_incoming)
    assert len(graph["historical_edges"]) == 7
    assert graph["external_citation_count"] == sum(
        edge["kind"] == "citation" and not edge["target_in_corpus"] for edge in edges
    )
    paper = read_item(db.get_container_client("papers"), target, target)
    assert paper["id"] == response["paper_id"]


def test_sync_reconciles_all_continuation_pages_and_duplicate_ids_across_partitions():
    from azure.cosmos import PartitionKey
    db = connect()
    container = db.create_container_if_not_exists("paging-probe", partition_key=PartitionKey("/paper_id"))
    try:
        desired = [{"id": str(i), "paper_id": str(i % 5), "value": i} for i in range(205)]
        desired.append({"id": "0", "paper_id": "different", "value": 206})
        assert sync_items(container, desired, "paper_id")["upserted"] == 206
        assert sync_items(container, desired, "paper_id") == {"upserted": 0, "deleted": 0, "unchanged": 206}
        changed = [{**desired[0], "value": -1}, *desired[101:]]
        assert sync_items(container, changed, "paper_id") == {"upserted": 1, "deleted": 100, "unchanged": 105}
        assert query(container, "SELECT VALUE COUNT(1) FROM c") == [106]
    finally:
        db.delete_container("paging-probe")


@pytest.mark.skipif(os.environ.get("REQUIRE_OFFLINE") != "1", reason="Set REQUIRE_OFFLINE=1 with the internal-network override")
def test_backend_cannot_access_external_services():
    import requests
    with pytest.raises(requests.RequestException):
        requests.get("https://huggingface.co", timeout=3)
    assert search("ASML EUCLIDES research program")["results"]
