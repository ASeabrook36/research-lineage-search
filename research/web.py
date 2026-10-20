import json
import logging
from pathlib import Path

from azure.core.exceptions import AzureError
from azure.cosmos.exceptions import CosmosResourceNotFoundError
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from research.collect import ROOT
from research.db import STORAGE_VERSION, connect, query, read_item, read_optional
from research.embed import CHUNK_VERSION, MODEL_PATH, MODEL_VERSION
from research.search import Filters, options, public, search, search_vector

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logging.getLogger("azure").setLevel(logging.WARNING)
log = logging.getLogger(__name__)
app = FastAPI(title="EUV Research Atlas", version="0.1.0", docs_url=None, redoc_url=None)
STATIC = Path(__file__).resolve().parent / "static"
DATA = Path(__file__).resolve().parent.parent / "data"


@app.exception_handler(AzureError)
def database_failure(request: Request, exc: AzureError):
    log.exception("Cosmos request failed: %s %s", request.method, request.url.path, exc_info=exc)
    return JSONResponse(status_code=503, content={
        "detail": "Local Cosmos DB is unavailable or rejected the query. Check docker compose logs and run the capability probe.",
    })


@app.exception_handler(RuntimeError)
def setup_failure(request: Request, exc: RuntimeError):
    log.exception("Local setup failure: %s", request.url.path, exc_info=exc)
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.middleware("http")
async def local_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    return response


@app.get("/api/status")
def status(live_counts: bool = False):
    db = connect()
    collection_path = ROOT / "collection-report.json"
    state = read_optional(db.get_container_client("settings"), "ingestion-state")
    report = state.get("report") if state else None
    complete = bool(
        state and state["status"] == "complete" and report
        and report.get("model_version") == MODEL_VERSION
        and report.get("chunk_version") == CHUNK_VERSION
        and report.get("storage_version") == STORAGE_VERSION
    )
    counts = {
        "papers": report["papers"] if complete else 0,
        "passages": report["passages"] if complete else 0,
        "edges": report["citation_edges"] + report["historical_edges"] if complete else 0,
    }
    if live_counts:
        counts = {
            name: query(db.get_container_client(name), "SELECT VALUE COUNT(1) FROM c")[0]
            for name in counts
        }
    collection = json.loads(collection_path.read_text()) if collection_path.exists() else None
    return {
        "ready": bool(complete and counts["passages"] and (MODEL_PATH / "modules.json").exists()),
        "counts": counts, "ingestion": report,
        "counts_source": "live" if live_counts else "completed_ingestion",
        "collection": {k: v for k, v in collection.items() if k != "failures"} if collection else None,
        "collection_failures": len(collection["failures"]) if collection else 0,
        "model_version": MODEL_VERSION,
    }


@app.get("/api/filters")
def filters():
    return options()


@app.get("/api/search")
def find(
    q: str = Query(min_length=1, max_length=600),
    country: list[str] = Query(default=[]), field: list[str] = Query(default=[]),
    institution: list[str] = Query(default=[]), author: list[str] = Query(default=[]),
    year_from: int | None = Query(default=None, ge=1980, le=2100),
    year_to: int | None = Query(default=None, ge=1980, le=2100),
    limit: int = Query(default=10, ge=1, le=30),
):
    try:
        return search(q, Filters(country, field, institution, author, year_from, year_to), limit)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@app.get("/api/papers/{paper_id}")
def paper_detail(paper_id: str):
    try:
        paper = public(read_item(connect().get_container_client("papers"), paper_id, paper_id))
    except CosmosResourceNotFoundError as exc:
        raise HTTPException(404, "Paper not found") from exc
    passages = query(
        connect().get_container_client("passages"),
        "SELECT c.id, c.text, c.page, c.section, c.source_url FROM c WHERE c.paper_id = @id",
        [{"name": "@id", "value": paper_id}],
        partition_key=paper_id,
    )
    passages.sort(key=lambda p: (p["section"] != "abstract", p["page"] or 0, p["id"]))
    return {"paper": paper, "passages": passages}


@app.get("/api/papers/{paper_id}/connections")
def connections(paper_id: str):
    db = connect()
    try:
        paper = read_item(db.get_container_client("papers"), paper_id, paper_id)
    except CosmosResourceNotFoundError as exc:
        raise HTTPException(404, "Paper not found") from exc
    outgoing = query(
        db.get_container_client("edges"),
        "SELECT * FROM c WHERE c.from_id = @id AND c.kind = 'citation'",
        [{"name": "@id", "value": paper_id}], partition_key=paper_id,
    )
    incoming = query(
        db.get_container_client("incoming-edges"),
        "SELECT * FROM c WHERE c.to_id = @id",
        [{"name": "@id", "value": paper_id}], partition_key=paper_id,
    )
    edges = {edge["id"]: edge for edge in outgoing + incoming}
    vectors = query(
        db.get_container_client("passages"),
        "SELECT TOP 1 c.embedding FROM c WHERE c.paper_id = @id",
        [{"name": "@id", "value": paper_id}],
        partition_key=paper_id,
    )
    if vectors:
        related, _ = search_vector(vectors[0]["embedding"], limit=5, exclude_paper=paper_id)
    else:
        related = []
    return {
        "paper_id": paper["id"], "citations": [public(e) for e in edges.values()],
        "related_research": related,
        "note": "Related research is semantic similarity only. Citation arrows run from citing paper to cited work.",
    }


@app.get("/api/lineage")
def lineage():
    history = json.loads((DATA / "history.json").read_text())
    container = connect().get_container_client("edges")
    in_corpus = query(container, "SELECT * FROM c WHERE c.kind = 'citation' AND c.target_in_corpus = true")
    historical = query(container, "SELECT * FROM c WHERE c.kind = 'historical'")
    external_count = query(
        container, "SELECT VALUE COUNT(1) FROM c WHERE c.kind = 'citation' AND c.target_in_corpus = false",
    )[0]
    return {
        "milestones": history["milestones"], "source_note": history["source_note"],
        "historical_edges": [public(e) for e in historical],
        "citations": [public(e) for e in in_corpus],
        "external_citation_count": external_count,
    }


app.mount("/", StaticFiles(directory=STATIC, html=True), name="site")
