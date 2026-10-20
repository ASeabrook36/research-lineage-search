import re

import pytest

from research.collect import abstract_text, doi_key, extract_pdf, merge_records, normalize_openalex, normalize_osti
from research.embed import chunks
from research.ingest import citation_edges, historical_edges
from research.search import Filters, filter_clause


def openalex(**overrides):
    return {
        "id": "https://openalex.org/W1", "title": "Soft X-Ray Imaging",
        "doi": "https://doi.org/10.1/example", "publication_year": 1991,
        "authorships": [], "abstract_inverted_index": {"Early": [0], "imaging": [1]},
        **overrides,
    }


def test_openalex_normalization_preserves_author_affiliations():
    work = openalex(authorships=[{
        "author": {"id": "A1", "display_name": "Researcher"},
        "institutions": [
            {"id": "I1", "display_name": "NTT", "country_code": "JP"},
            {"id": "I2", "display_name": "US Lab", "country_code": "US"},
        ],
    }])
    result = normalize_openalex(work)
    assert result["countries"] == ["JP", "US"]
    assert result["authors"][0]["institution_ids"] == ["I1", "I2"]
    assert result["abstract"] == "Early imaging"


def test_missing_country_is_unknown_not_inferred_from_publication():
    result = normalize_osti({
        "osti_id": "1", "title": "Example", "publication_date": "1997-01-01",
        "country_publication": "United States", "authors": ["An Author"],
    })
    assert result["countries"] == ["Unknown"]
    assert normalize_openalex(openalex())["countries"] == ["Unknown"]


def test_osti_affiliation_mapping():
    result = normalize_osti({
        "osti_id": "1", "title": "Example", "publication_date": "1997-01-01",
        "authors": ["A. Author [Sandia (United States)]"],
    })
    assert result["countries"] == ["US"]
    assert result["authors"][0]["institution_ids"] == [result["institutions"][0]["id"]]


def test_abstract_positions_are_reconstructed():
    assert abstract_text({"three": [2], "one": [0], "two": [1]}) == "one two three"
    assert abstract_text(None) == ""
    assert doi_key("HTTPS://doi.org/10.1/A") == "10.1/a"


def test_deduplication_merges_aliases_and_pdf_links():
    first = normalize_openalex(openalex())
    second = normalize_osti({
        "osti_id": "99", "title": first["title"], "doi": "10.1/example",
        "publication_date": "1991-01-01", "links": [{"rel": "fulltext", "href": "https://www.osti.gov/servlets/purl/99"}],
    })
    merged = merge_records([first, second])
    assert len(merged) == 1
    assert merged[0]["aliases"] == ["https://openalex.org/W1", "osti:99"]
    assert len(merged[0]["sources"]) == 2
    assert len(merged[0]["pdf_urls"]) == 1


def test_distinct_dois_are_not_fuzzily_merged():
    first = normalize_openalex(openalex())
    second = normalize_openalex(openalex(id="W2", doi="10.1/different"))
    assert len(merge_records([first, second])) == 2


def test_generic_title_without_doi_requires_author_overlap():
    first = normalize_openalex(openalex(doi=None))
    second = normalize_openalex(openalex(id="W2", doi=None))
    assert len(merge_records([first, second])) == 2


def test_filter_values_are_parameters_not_sql():
    clauses, parameters = filter_clause(Filters(country=["JP", "US"], author=["' OR 1=1 --"], year_from=1986, year_to=2006))
    sql = " AND ".join(clauses)
    assert " OR " in clauses[0]
    assert "authors" in sql
    assert "' OR 1=1 --" not in sql
    assert parameters[2]["value"] == "' OR 1=1 --"
    assert "c.year >= @year_from" in sql


def test_invalid_filters_fail_explicitly():
    with pytest.raises(ValueError, match="Start year"):
        filter_clause(Filters(year_from=2000, year_to=1990))
    with pytest.raises(ValueError, match="At most 30"):
        filter_clause(Filters(country=["JP"] * 31))


class Tokenizer:
    def encode(self, text, add_special_tokens=True):
        return list(range(len(text.split()) + (2 if add_special_tokens else 0)))

    def decode(self, ids):
        return " ".join("title" for _ in ids)

    def __call__(self, text, **kwargs):
        return {"offset_mapping": [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]}


def test_chunks_preserve_verbatim_passages_and_pages():
    paper = normalize_openalex(openalex())
    paper.update(coverage="full_text", pages=[{"page": 3, "text": " ".join(f"word{i}" for i in range(450))}])
    result = chunks(paper, Tokenizer())
    assert all(len(Tokenizer().encode(p["embedding_text"])) <= 256 for p in result)
    full_text = [p for p in result if p["section"] == "full_text"]
    assert len(full_text) == 3
    assert all(p["page"] == 3 for p in full_text)
    assert all(p["text"] in paper["pages"][0]["text"] for p in full_text)
    assert full_text[0]["text"].endswith("word199")
    assert full_text[1]["text"].startswith("word168")
    assert len({p["id"] for p in result}) == len(result)


def test_metadata_only_does_not_fabricate_passages():
    paper = normalize_openalex(openalex(abstract_inverted_index=None))
    paper.update(coverage="metadata", pages=[])
    assert chunks(paper, Tokenizer()) == []


def test_citations_have_direction_and_external_labels():
    first = normalize_openalex(openalex(referenced_works=["W2", "W3"]))
    second = normalize_openalex(openalex(id="W2", doi="10.1/second"))
    edges = citation_edges([first, second])
    assert edges[0]["from_id"] == first["id"]
    assert edges[0]["to_id"] == second["id"]
    assert edges[0]["target_in_corpus"] is True
    assert edges[1]["to_id"] == "W3"
    assert edges[1]["target_in_corpus"] is False


def test_history_is_attributed_and_never_semantic_influence():
    for edge in historical_edges():
        assert edge["kind"] == "historical"
        assert edge["evidence_url"].startswith("https://")
        assert edge["relationship"] in {"historical_sequence", "documented_collaboration", "documented_program_connection"}
        assert edge["explanation"]


def test_invalid_pdf_is_not_accepted():
    from pypdf.errors import PyPdfError
    with pytest.raises(PyPdfError):
        extract_pdf(b"not a PDF")


def test_disclaimer_only_pdf_is_not_mistaken_for_research(monkeypatch):
    class Page:
        def extract_text(self):
            return "DISCLAIMER This document was prepared as an account of work sponsored by the United States Government. " * 20

    class Reader:
        is_encrypted = False
        pages = [Page()]

    monkeypatch.setattr("research.collect.PdfReader", lambda payload: Reader())
    with pytest.raises(ValueError, match="insufficient"):
        extract_pdf(b"%PDF")


def test_cloud_endpoints_are_rejected_before_connecting(monkeypatch):
    from research.db import client
    for endpoint in ["https://example.documents.azure.com", "http://localhost:8081@evil.example"]:
        client.cache_clear()
        monkeypatch.setenv("COSMOS_ENDPOINT", endpoint)
        with pytest.raises(ValueError, match="Only the local"):
            client()
    client.cache_clear()
