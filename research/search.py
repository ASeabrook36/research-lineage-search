import time
from dataclasses import dataclass, field as dc_field

from research.db import connect, query
from research.embed import MODEL_VERSION, embed


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
    return {key: value for key, value in item.items() if not key.startswith("_")}


def papers():
    return [public(p) for p in query(connect().get_container_client("papers"), "SELECT * FROM c")]


def options():
    records = papers()
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


def search_vector(vector, filters=None, limit=10, exclude_paper=None):
    if not 1 <= limit <= 30:
        raise ValueError("Result limit must be between 1 and 30")
    settings = connect().get_container_client("settings")
    state = query(settings, "SELECT * FROM c WHERE c.id = 'ingestion-state'")
    if not state or state[0]["status"] != "complete":
        raise RuntimeError("Research ingestion is missing or incomplete. Finish: python -m research ingest")
    if state[0]["report"]["model_version"] != MODEL_VERSION:
        raise RuntimeError("Stored embeddings do not match the local model version")
    filters = filters or Filters()
    clauses, parameters = filter_clause(filters)
    clauses.append("c.model_version = @model")
    parameters += [{"name": "@model", "value": MODEL_VERSION}, {"name": "@vector", "value": vector}]
    if exclude_paper:
        clauses.append("c.paper_id != @exclude")
        parameters.append({"name": "@exclude", "value": exclude_paper})
    where = " WHERE " + " AND ".join(clauses)
    projection = "c.id, c.paper_id, c.text, c.title, c.year, c.countries, c.page, c.section, c.source_url, c.coverage"
    container = connect().get_container_client("passages")
    candidate_limit = min(2500, max(40, limit * 8))
    while True:
        rows = query(
            container,
            f"SELECT TOP {candidate_limit} {projection}, VectorDistance(c.embedding, @vector) AS score FROM c"
            + where + " ORDER BY VectorDistance(c.embedding, @vector)",
            parameters,
        )
        groups = {}
        for row in rows:
            if row["paper_id"] not in groups:
                groups[row["paper_id"]] = {
                    "id": row["paper_id"], "title": row["title"], "year": row["year"],
                    "countries": row["countries"], "coverage": row["coverage"],
                    "score": row["score"], "passages": [],
                }
            if len(groups[row["paper_id"]]["passages"]) < 2:
                groups[row["paper_id"]]["passages"].append(row)
        if len(groups) >= limit or len(rows) < candidate_limit or candidate_limit == 2500:
            return list(groups.values())[:limit], len(rows)
        candidate_limit = min(2500, candidate_limit * 3)


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
