import logging
import time
from dataclasses import dataclass, field as dc_field

from research.db import STORAGE_VERSION, RequestMetrics, connect, query, read_optional
from research.embed import CHUNK_VERSION, MODEL_VERSION, embed

log = logging.getLogger(__name__)


@dataclass
class Filters:
    country: list[str] = dc_field(default_factory=list)
    field: list[str] = dc_field(default_factory=list)
    institution: list[str] = dc_field(default_factory=list)
    author: list[str] = dc_field(default_factory=list)
    year_from: int | None = None
    year_to: int | None = None


def filter_clause(filters):
    clauses, parameters = [], []
    for category, property_ in (
        ("country", "countries"), ("field", "fields"),
        ("institution", "institutions"), ("author", "authors"),
    ):
        values = getattr(filters, category)
        if len(values) > 30:
            raise ValueError(f"At most 30 selections allowed for {category}")
        choices = []
        for index, value in enumerate(dict.fromkeys(values)):
            name = f"@{category}{index}"
            parameters.append({"name": name, "value": value})
            choices.append(f"ARRAY_CONTAINS(c.{property_}, {name})")
        if choices:
            clauses.append("(" + " OR ".join(choices) + ")")
    if filters.year_from is not None:
        clauses.append("c.year >= @year_from")
        parameters.append({"name": "@year_from", "value": filters.year_from})
    if filters.year_to is not None:
        clauses.append("c.year <= @year_to")
        parameters.append({"name": "@year_to", "value": filters.year_to})
    if filters.year_from is not None and filters.year_to is not None and filters.year_from > filters.year_to:
        raise ValueError("Start year must not exceed end year")
    return clauses, parameters


def public(item):
    return {key: value for key, value in item.items() if not key.startswith("_") and key != "sync_hash"}


def papers():
    return [public(p) for p in query(connect().get_container_client("papers"), "SELECT * FROM c")]


def build_options(records):
    countries, fields, institutions, authors = {}, {}, {}, {}
    for paper in records:
        for key, target in (("countries", countries), ("fields", fields)):
            for value in set(paper[key]):
                target[value] = target.get(value, 0) + 1
        for key, target in (("institutions", institutions), ("authors", authors)):
            for value in paper[key]:
                target[value["id"]] = value["name"]
            if not paper[key]:
                target["Unknown"] = "Unknown"
    return {
        "country": [{"id": k, "name": k, "count": v} for k, v in sorted(countries.items())],
        "field": [{"id": k, "name": k, "count": v} for k, v in sorted(fields.items())],
        "institution": sorted([{"id": k, "name": v} for k, v in institutions.items()], key=lambda x: x["name"]),
        "author": sorted([{"id": k, "name": v} for k, v in authors.items()], key=lambda x: x["name"]),
    }


def require_ingestion():
    state = read_optional(connect().get_container_client("settings"), "ingestion-state")
    if not state or state["status"] != "complete":
        raise RuntimeError("Research ingestion is missing or incomplete. Finish: python -m research ingest")
    report = state["report"]
    if report["model_version"] != MODEL_VERSION or report.get("chunk_version") != CHUNK_VERSION:
        raise RuntimeError("Stored embeddings do not match the local model/chunk version")
    if report.get("storage_version") != STORAGE_VERSION:
        raise RuntimeError("Stored Cosmos schema is incompatible; use the documented storage migration")
    return state


def options():
    state = require_ingestion()
    stored = read_optional(connect().get_container_client("settings"), "filter-options")
    if not stored or stored["ingested_at"] != state["report"]["ingested_at"]:
        raise RuntimeError("Corpus filter options are missing or incomplete. Run: python -m research ingest")
    return stored["options"]


def search_vector(vector, filters=None, limit=10, exclude_paper=None):
    if not 1 <= limit <= 30:
        raise ValueError("Result limit must be between 1 and 30")
    state = require_ingestion()
    filters = filters or Filters()
    clauses, parameters = filter_clause(filters)
    clauses.append("c.model_version = @model")
    parameters += [{"name": "@model", "value": MODEL_VERSION}, {"name": "@vector", "value": vector}]
    if exclude_paper:
        clauses.append("c.paper_id != @exclude")
        parameters.append({"name": "@exclude", "value": exclude_paper})
    where = " WHERE " + " AND ".join(clauses)
    container = connect().get_container_client("passages")
    cap = state["report"]["passages"]
    candidate_limit = min(cap, max(40, limit * 8))
    if candidate_limit < 1:
        raise RuntimeError("Completed ingestion contains no searchable passages")
    metrics = RequestMetrics()
    attempts, total_candidates, hydration_queries = 0, 0, 0
    fields = ("id", "paper_id", "text", "title", "year", "countries", "page", "section", "source_url", "coverage")
    start = time.perf_counter()
    while True:
        include_evidence = attempts == 0
        projection = ", ".join(f"c.{key}" for key in fields) if include_evidence else "c.id, c.paper_id"
        rows = query(
            container,
            f"SELECT TOP {candidate_limit} {projection}, VectorDistance(c.embedding, @vector) AS score FROM c"
            + where + " ORDER BY VectorDistance(c.embedding, @vector)",
            parameters, metrics=metrics,
        )
        attempts += 1
        total_candidates += len(rows)
        groups = {}
        for row in rows:
            group = groups.setdefault(row["paper_id"], [])
            if len(group) < 2:
                group.append(row)
        if len(groups) >= limit or len(rows) < candidate_limit or candidate_limit == cap:
            break
        candidate_limit = min(cap, candidate_limit * 3)
    results = []
    for paper_id, candidates in list(groups.items())[:limit]:
        if include_evidence:
            documents = {item["id"]: item for item in candidates}
        else:
            id_parameters = [{"name": f"@id{i}", "value": c["id"]} for i, c in enumerate(candidates)]
            selected = query(
                container,
                "SELECT " + ", ".join(f"c.{key}" for key in fields)
                + " FROM c WHERE c.paper_id = @paper AND c.id IN ("
                + ", ".join(p["name"] for p in id_parameters) + ")",
                [{"name": "@paper", "value": paper_id}, *id_parameters],
                partition_key=paper_id, metrics=metrics,
            )
            hydration_queries += 1
            documents = {item["id"]: item for item in selected}
            if len(documents) != len(candidates) or any(c["id"] not in documents for c in candidates):
                raise RuntimeError("Research passages changed during retrieval. Wait for ingestion to complete")
        passages = []
        for candidate in candidates:
            item = documents[candidate["id"]]
            passages.append({**{key: item[key] for key in fields}, "score": candidate["score"]})
        first = passages[0]
        results.append({
            "id": paper_id, "title": first["title"], "year": first["year"],
            "countries": first["countries"], "coverage": first["coverage"],
            "score": first["score"], "passages": passages,
        })
    log.info(
        "Cosmos vector search attempts=%d total_candidates=%d final_candidates=%d hydrated=%d hydration_queries=%d elapsed_ms=%.1f metrics=%s",
        attempts, total_candidates, len(rows), sum(len(r["passages"]) for r in results),
        hydration_queries, (time.perf_counter() - start) * 1000, metrics.summary(),
    )
    return results, len(rows)


def search(text, filters=None, limit=10):
    if not text.strip():
        raise ValueError("Enter a topic or research question")
    if len(text) > 600:
        raise ValueError("Search text must be at most 600 characters")
    # Do not silently truncate a long question at the embedding model boundary.
    from research.embed import model
    if len(model().tokenizer.encode(text)) > 256:
        raise ValueError("Question exceeds the embedding model's 256-token limit; shorten it")
    start = time.perf_counter()
    results, candidates = search_vector(embed([text])[0], filters, limit)
    return {
        "query": text, "results": results, "candidate_passages": candidates,
        "elapsed_ms": round((time.perf_counter() - start) * 1000),
        "engine": "Cosmos DB VectorDistance", "model_version": MODEL_VERSION,
        "score_note": "The tested emulator returns cosine similarity (higher is more similar). Scores are not probabilities or evidence of influence.",
    }
