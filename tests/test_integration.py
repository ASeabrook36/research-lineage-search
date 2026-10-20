import os

import numpy as np
import pytest
from azure.cosmos.exceptions import CosmosHttpResponseError

from research.db import connect, probe, query
from research.embed import embed, model
from research.search import Filters, filter_clause, papers, search, search_vector
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
        sync_items(container, desired, "paper_id")
        sync_items(container, desired, "paper_id")
        rows = query(container, "SELECT c.id, c.updated FROM c")
        assert {row["id"] for row in rows} == {"new", "keep"}
        assert next(row for row in rows if row["id"] == "keep")["updated"] is True
    finally:
        db.delete_container("idempotency-probe")


@pytest.mark.skipif(os.environ.get("REQUIRE_OFFLINE") != "1", reason="Set REQUIRE_OFFLINE=1 with the internal-network override")
def test_backend_cannot_access_external_services():
    import requests
    with pytest.raises(requests.RequestException):
        requests.get("https://huggingface.co", timeout=3)
    assert search("ASML EUCLIDES research program")["results"]
