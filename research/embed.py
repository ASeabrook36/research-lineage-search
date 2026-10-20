from functools import lru_cache

from sentence_transformers import SentenceTransformer

from research.collect import ROOT, digest

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
MODEL_VERSION = MODEL_ID + "@" + MODEL_REVISION
CHUNK_VERSION = "token-window-200-overlap32-title40-v1"
MODEL_PATH = ROOT / "models" / "minilm"


@lru_cache(maxsize=1)
def model():
    if not (MODEL_PATH / "modules.json").exists():
        raise RuntimeError("Local embedding model is missing. Run: python -m research ingest")
    return SentenceTransformer(str(MODEL_PATH), device="cpu", local_files_only=True)


def prepare_model():
    if not (MODEL_PATH / "modules.json").exists():
        downloaded = SentenceTransformer(MODEL_ID, revision=MODEL_REVISION, device="cpu")
        downloaded.save(str(MODEL_PATH))
    return model()


def embed(texts):
    return model().encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False).tolist()


def chunks(paper, tokenizer):
    title_ids = tokenizer.encode(paper["title"], add_special_tokens=False)[:40]
    prefix = tokenizer.decode(title_ids) + "\n"
    sections = ([{"page": None, "text": paper["abstract"], "section": "abstract"}] if paper["abstract"] else [])
    sections += [{**page, "section": "full_text"} for page in paper.get("pages", [])]
    output = []
    for section in sections:
        text = section["text"]
        tokens = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        offsets = tokens["offset_mapping"]
        for start in range(0, len(offsets), 168):
            window = offsets[start:start + 200]
            if not window:
                continue
            passage = text[window[0][0]:window[-1][1]].strip()
            if len(passage) < 40:
                continue
            embedding_text = prefix + passage
            if len(tokenizer.encode(embedding_text)) > 256:
                raise ValueError("Chunk exceeds embedding model token limit")
            output.append({
                "id": digest(paper["id"] + str(section["page"]) + section["section"] + str(start) + passage),
                "paper_id": paper["id"], "text": passage, "embedding_text": embedding_text,
                "page": section["page"], "section": section["section"],
                "source_url": paper.get("full_text_url", paper["source_url"]) if section["section"] == "full_text" else paper["source_url"],
                "model_version": MODEL_VERSION, "chunk_version": CHUNK_VERSION,
                "title": paper["title"], "year": paper["year"], "coverage": paper["coverage"],
                "countries": paper["countries"], "fields": paper["fields"],
                "institutions": [i["id"] for i in paper["institutions"]] or ["Unknown"],
                "authors": [a["id"] for a in paper["authors"]] or ["Unknown"],
            })
            if start + 200 >= len(offsets):
                break
    return output
