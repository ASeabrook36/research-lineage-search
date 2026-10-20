import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from research.collect import ROOT, digest, write_json
from research.db import INDEX_POLICY, VECTOR_POLICY, connect, query
from research.embed import CHUNK_VERSION, MODEL_VERSION, chunks, embed, prepare_model

log = logging.getLogger(__name__)


def sync_items(container, items, partition_field):
    desired = {(item["id"], item[partition_field]) for item in items}
    for item in items:
        container.upsert_item(item)
    projection = "c.id" if partition_field == "id" else f"c.id, c.{partition_field}"
    existing = query(container, f"SELECT {projection} FROM c")
    for item in existing:
        if (item["id"], item[partition_field]) not in desired:
            container.delete_item(item["id"], partition_key=item[partition_field])


def citation_edges(papers):
    aliases = {alias: paper for paper in papers for alias in paper["aliases"]}
    edges = []
    for paper in papers:
        for reference in paper["references"]:
            target = aliases.get(reference)
            edges.append({
                "id": digest(paper["id"] + reference), "kind": "citation",
                "from_id": paper["id"], "to_id": target["id"] if target else reference,
                "from_title": paper["title"], "to_title": target["title"] if target else reference.rsplit("/", 1)[-1],
                "year": paper["year"], "target_in_corpus": bool(target),
                "evidence_url": next((s["url"] for s in paper["sources"] if s["provider"] == "OpenAlex"), paper["source_url"]),
                "explanation": "The source metadata records that this paper cites the target. A citation is not proof of technology transfer.",
            })
    return edges


def historical_edges():
    path = Path(__file__).resolve().parent.parent / "data" / "history.json"
    history = json.loads(path.read_text())
    return [{**edge, "kind": "historical", "id": digest(json.dumps(edge, sort_keys=True))} for edge in history["edges"]]


def ingest():
    import fcntl
    ROOT.mkdir(parents=True, exist_ok=True)
    with (ROOT / "ingestion.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another ingestion is active; wait for it to finish") from exc
        return ingest_locked()


def ingest_locked():
    corpus_path = ROOT / "corpus.json"
    if not corpus_path.exists():
        raise RuntimeError("Corpus is missing. Run: python -m research collect")
    papers = json.loads(corpus_path.read_text())
    local_model = prepare_model()
    db = connect(True)
    seeds = json.loads((Path(__file__).resolve().parent.parent / "data" / "seeds.json").read_text())
    passage_sets = {paper["id"]: chunks(paper, local_model.tokenizer) for paper in papers}
    # Round-robin allocation prevents long reports from crowding out other papers.
    passages = []
    for index in range(max((len(items) for items in passage_sets.values()), default=0)):
        for items in passage_sets.values():
            if index < len(items):
                passages.append(items[index])
                if len(passages) >= seeds["max_passages"]:
                    break
        if len(passages) >= seeds["max_passages"]:
            break
    cache_path = ROOT / "embedding-cache.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    pending = [p for p in passages if digest(MODEL_VERSION + p["embedding_text"]) not in cache]
    log.info("Embedding %d new passages (%d total)", len(pending), len(passages))
    for start in range(0, len(pending), 64):
        batch = pending[start:start + 64]
        for passage, vector in zip(batch, embed([p["embedding_text"] for p in batch]), strict=True):
            cache[digest(MODEL_VERSION + passage["embedding_text"])] = vector
        write_json(cache_path, cache)
        log.info("Embedded %d/%d", min(start + 64, len(pending)), len(pending))
    for passage in passages:
        passage["embedding"] = cache[digest(MODEL_VERSION + passage.pop("embedding_text"))]
    if not passages:
        raise RuntimeError("No genuine research passages available; ingestion refused")
    counts = {}
    for passage in passages:
        counts[passage["paper_id"]] = counts.get(passage["paper_id"], 0) + 1
    # Check compatibility before writing; policy changes require explicit container recreation.
    properties = db.get_container_client("passages").read()
    policy = properties.get("vectorEmbeddingPolicy")
    if policy is not None and policy != VECTOR_POLICY:
        raise RuntimeError("Incompatible Cosmos vector policy; see documented reset procedure")
    if properties.get("indexingPolicy", {}).get("vectorIndexes") != INDEX_POLICY["vectorIndexes"]:
        raise RuntimeError("Cosmos did not retain the requested DiskANN index policy")
    if policy is None:
        log.warning("Emulator omits vectorEmbeddingPolicy on container reads; known-vector probe verifies behavior, not policy introspection")
    settings = db.get_container_client("settings")
    stored = query(settings, "SELECT * FROM c WHERE c.id = 'embedding-config'")
    config = {"id": "embedding-config", "model_version": MODEL_VERSION, "chunk_version": CHUNK_VERSION, "vector_policy": VECTOR_POLICY}
    if stored and any(stored[0].get(key) != value for key, value in config.items()):
        raise RuntimeError("Stored embedding configuration differs; explicitly reset before changing models")
    settings.upsert_item(config)
    settings.upsert_item({"id": "ingestion-state", "status": "running"})
    sync_items(db.get_container_client("passages"), passages, "paper_id")
    for paper in papers:
        paper["passage_count"] = counts.get(paper["id"], 0)
        paper["available_passages"] = len(passage_sets[paper["id"]])
        paper["extracted_page_count"] = len(paper.get("pages", []))
    sync_items(db.get_container_client("papers"), [
        {key: value for key, value in paper.items() if key != "pages"} for paper in papers
    ], "id")
    edges = citation_edges(papers) + historical_edges()
    sync_items(db.get_container_client("edges"), edges, "kind")
    report = {
        "ingested_at": datetime.now(timezone.utc).isoformat(),
        "papers": len(papers), "passages": len(passages),
        "full_text_papers": sum(p["coverage"] == "full_text" for p in papers),
        "metadata_only_papers": sum(p["coverage"] == "metadata" for p in papers),
        "citation_edges": sum(e["kind"] == "citation" for e in edges),
        "in_corpus_citations": sum(e["kind"] == "citation" and e["target_in_corpus"] for e in edges),
        "historical_edges": sum(e["kind"] == "historical" for e in edges),
        "model_version": MODEL_VERSION, "chunk_version": CHUNK_VERSION,
        "passage_cap": seeds["max_passages"],
        "truncated_papers": sum(p["passage_count"] < p["available_passages"] for p in papers),
        "index_note": "DiskANN configured; index acceleration and cloud RU behavior are not established by this emulator.",
    }
    write_json(ROOT / "ingestion-report.json", report)
    settings.upsert_item({"id": "ingestion-state", "status": "complete", "report": report})
    log.info("Ingestion complete: %s", report)
    return report
