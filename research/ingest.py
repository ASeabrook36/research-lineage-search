import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

from research.collect import ROOT, digest, write_json
from research.db import STORAGE_VERSION, VECTOR_POLICY, RequestMetrics, connect, iter_query, read_optional
from research.embed import CHUNK_VERSION, MODEL_VERSION, chunks, embed, prepare_model
from research.search import build_options

log = logging.getLogger(__name__)


def sync_items(container, items, partition_field):
    start = time.perf_counter()
    desired = {}
    for source in items:
        item = {key: value for key, value in source.items() if not key.startswith("_") and key != "sync_hash"}
        content_hash = hashlib.sha256(json.dumps(item, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        identity = (item["id"], item[partition_field])
        if identity in desired:
            raise ValueError(f"Duplicate document identity in {container.id}: {identity}")
        desired[identity] = {**item, "sync_hash": content_hash}
    projection = "c.id, c.sync_hash" + ("" if partition_field == "id" else f", c.{partition_field}")
    metrics = RequestMetrics()
    existing = {
        (item["id"], item[partition_field]): item.get("sync_hash")
        for item in iter_query(container, f"SELECT {projection} FROM c", metrics=metrics)
    }
    counts = {"upserted": 0, "deleted": 0, "unchanged": 0}
    for identity, item in desired.items():
        if existing.get(identity) == item["sync_hash"]:
            counts["unchanged"] += 1
        else:
            container.upsert_item(item, response_hook=metrics.record)
            counts["upserted"] += 1
    for item_id, partition_key in existing.keys() - desired.keys():
        container.delete_item(item_id, partition_key=partition_key, response_hook=metrics.record)
        counts["deleted"] += 1
    log.info(
        "Cosmos sync container=%s counts=%s elapsed_ms=%.1f metrics=%s",
        container.id, counts, (time.perf_counter() - start) * 1000, metrics.summary(),
    )
    return counts


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
    settings = db.get_container_client("settings")
    stored = read_optional(settings, "embedding-config")
    config = {"id": "embedding-config", "model_version": MODEL_VERSION, "chunk_version": CHUNK_VERSION, "vector_policy": VECTOR_POLICY}
    if stored and any(stored.get(key) != value for key, value in config.items()):
        raise RuntimeError("Stored embedding configuration differs; explicitly reset before changing models")
    if not stored:
        settings.upsert_item(config)
    settings.upsert_item({"id": "ingestion-state", "status": "running"})
    sync = {"passages": sync_items(db.get_container_client("passages"), passages, "paper_id")}
    for paper in papers:
        paper["passage_count"] = counts.get(paper["id"], 0)
        paper["available_passages"] = len(passage_sets[paper["id"]])
        paper["extracted_page_count"] = len(paper.get("pages", []))
    sync["papers"] = sync_items(db.get_container_client("papers"), [
        {key: value for key, value in paper.items() if key != "pages"} for paper in papers
    ], "id")
    edges = citation_edges(papers) + historical_edges()
    sync["edges"] = sync_items(db.get_container_client("edges"), edges, "from_id")
    sync["incoming-edges"] = sync_items(
        db.get_container_client("incoming-edges"),
        [edge for edge in edges if edge["kind"] == "citation" and edge["target_in_corpus"]], "to_id",
    )
    report = {
        "ingested_at": datetime.now(timezone.utc).isoformat(),
        "papers": len(papers), "passages": len(passages),
        "full_text_papers": sum(p["coverage"] == "full_text" for p in papers),
        "metadata_only_papers": sum(p["coverage"] == "metadata" for p in papers),
        "citation_edges": sum(e["kind"] == "citation" for e in edges),
        "in_corpus_citations": sum(e["kind"] == "citation" and e["target_in_corpus"] for e in edges),
        "historical_edges": sum(e["kind"] == "historical" for e in edges),
        "model_version": MODEL_VERSION, "chunk_version": CHUNK_VERSION,
        "storage_version": STORAGE_VERSION, "sync": sync,
        "passage_cap": seeds["max_passages"],
        "truncated_papers": sum(p["passage_count"] < p["available_passages"] for p in papers),
        "index_note": "DiskANN configured; index acceleration and cloud RU behavior are not established by this emulator.",
    }
    settings.upsert_item({"id": "filter-options", "ingested_at": report["ingested_at"], "options": build_options(papers)})
    write_json(ROOT / "ingestion-report.json", report)
    settings.upsert_item({"id": "ingestion-state", "status": "complete", "report": report})
    log.info("Ingestion complete: %s", report)
    return report
