import hashlib
import io
import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import requests
from pypdf import PdfReader
from pypdf.errors import PyPdfError

log = logging.getLogger(__name__)
ROOT = Path(os.environ.get("DATA_DIR", ".local"))
SEEDS = Path(__file__).resolve().parent.parent / "data" / "seeds.json"
SESSION = requests.Session()
SESSION.headers["User-Agent"] = "EUVResearchPrototype/0.1 (local academic discovery)"


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()[:24]


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temporary.replace(path)


def clean(value):
    return re.sub(r"\s+", " ", value or "").strip()


def doi_key(value):
    return re.sub(r"^https?://(?:dx\.)?doi\.org/", "", value or "", flags=re.IGNORECASE).lower().strip()


def stable_id(doi, fallback):
    return digest(doi_key(doi) or fallback)


def abstract_text(index):
    if not index:
        return ""
    positions = {position: word for word, places in index.items() for position in places}
    return " ".join(positions[p] for p in sorted(positions))


def normalize_openalex(work):
    authors = []
    institutions = {}
    countries = set()
    for authorship in work.get("authorships", []):
        affiliation_ids = []
        for institution in authorship.get("institutions", []):
            key = institution.get("id") or institution["display_name"]
            institutions[key] = {
                "id": key, "name": institution["display_name"],
                "country": institution.get("country_code") or "Unknown",
            }
            affiliation_ids.append(key)
            if institution.get("country_code"):
                countries.add(institution["country_code"])
        author = authorship["author"]
        authors.append({
            "id": author.get("id") or author["display_name"],
            "name": author["display_name"], "institution_ids": affiliation_ids,
        })
    topic = work.get("primary_topic") or {}
    fields = sorted({
        x["display_name"] for x in [topic, topic.get("subfield", {}), topic.get("field", {})]
        if x.get("display_name")
    })
    locations = work.get("locations", []) + [work.get("best_oa_location") or {}]
    pdfs = sorted({
        location["pdf_url"] for location in locations
        if location and location.get("pdf_url") and location.get("is_oa")
    })
    return {
        "id": stable_id(work.get("doi"), work["id"]),
        "aliases": [work["id"]], "doi": doi_key(work.get("doi")),
        "title": clean(work["title"]), "year": work.get("publication_year"),
        "authors": authors, "institutions": list(institutions.values()),
        "countries": sorted(countries) or ["Unknown"], "fields": fields or ["Unknown"],
        "abstract": abstract_text(work.get("abstract_inverted_index")),
        "references": work.get("referenced_works", []),
        "source_url": work.get("doi") or work["id"],
        "sources": [{"provider": "OpenAlex", "url": work["id"]}],
        "pdf_urls": pdfs,
        "licenses": sorted({str(loc["license"]) for loc in locations if loc and loc.get("license")}),
        "kind": "research", "record_type": work.get("type", "research"),
    }


def normalize_osti(work):
    authors = []
    institutions = {}
    for entry in work.get("authors", []):
        name = clean(entry.split("[")[0].split("(ORCID")[0])
        match = re.search(r"\[(.+?)\]", entry)
        ids = []
        if match:
            for affiliation in match.group(1).split(";"):
                affiliation = clean(affiliation)
                key = "osti-institution-" + digest(affiliation)
                country = "US" if "(United States)" in affiliation else "Unknown"
                institutions[key] = {"id": key, "name": affiliation, "country": country}
                ids.append(key)
        authors.append({"id": "osti-author-" + digest(name), "name": name, "institution_ids": ids})
    links = work.get("links", [])
    return {
        "id": stable_id(work.get("doi"), "osti:" + work["osti_id"]),
        "aliases": ["osti:" + work["osti_id"]], "doi": doi_key(work.get("doi")),
        "title": clean(work.get("title")),
        "year": int(work["publication_date"][:4]) if work.get("publication_date") else None,
        "authors": authors, "institutions": list(institutions.values()),
        "countries": sorted({i["country"] for i in institutions.values()}) or ["Unknown"],
        "fields": work.get("subjects") or ["Unknown"],
        "abstract": clean(work.get("description")), "references": [],
        "source_url": "https://www.osti.gov/biblio/" + work["osti_id"],
        "sources": [{"provider": "OSTI", "url": "https://www.osti.gov/biblio/" + work["osti_id"]}],
        "pdf_urls": [link["href"] for link in links if link["rel"] == "fulltext"],
        "licenses": [], "kind": "research", "record_type": work.get("product_type", "research"),
    }


def merge_records(records):
    merged = {}
    titles = {}
    for record in records:
        title = re.sub(r"[^a-z0-9]", "", record["title"].lower())
        title_key = (title, record["year"])
        existing_id = record["id"] if record["id"] in merged else titles.get(title_key)
        if existing_id:
            existing = merged[existing_id]
            # Title/year matching is used only when at least one DOI is missing.
            if existing["doi"] and record["doi"] and existing["doi"] != record["doi"]:
                existing_id = None
            elif existing["id"] != record["id"]:
                def author_keys(item):
                    return {
                        " ".join(sorted(re.findall(r"[a-z]+", a["name"].lower())))
                        for a in item["authors"]
                    }
                if not author_keys(existing).intersection(author_keys(record)):
                    existing_id = None
        if not existing_id:
            merged[record["id"]] = record
            titles[title_key] = record["id"]
            continue
        existing = merged[existing_id]
        for key in ("aliases", "references", "pdf_urls", "licenses"):
            existing[key] = sorted(set(existing[key] + record[key]))
        existing["sources"] += [s for s in record["sources"] if s not in existing["sources"]]
        if not existing["abstract"]:
            existing["abstract"] = record["abstract"]
        if not existing["institutions"] and record["institutions"]:
            for key in ("authors", "institutions", "countries"):
                existing[key] = record[key]
    return list(merged.values())


def cached_json(url, params):
    cache = ROOT / "raw"
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / (digest(url + json.dumps(params, sort_keys=True)) + ".json")
    if path.exists():
        return json.loads(path.read_text())
    for attempt in range(4):
        time.sleep(0.3 + attempt)
        response = SESSION.get(url, params=params, timeout=(10, 60))
        if response.status_code in (429, 500, 502, 503, 504) and attempt < 3:
            continue
        response.raise_for_status()
        result = response.json()
        write_json(path, result)
        return result
    raise RuntimeError("API retry loop exhausted")


def download_pdf(url):
    permitted = (
        "osti.gov", "arxiv.org", "jstage.jst.go.jp", "tudelft.nl", "utwente.nl",
        "escholarship.org", "lbl.gov", "llnl.gov", "sandia.gov", "mpg.de",
        "researchgate.net", "intechopen.com", "springeropen.com", "nature.com",
        "iop.org", "aps.org", "spiedigitallibrary.org", "aip.org", "osa.org",
        "optica.org", "pubs.acs.org", "pmc.ncbi.nlm.nih.gov",
    )
    for _ in range(6):
        parsed = urlparse(url)
        host = (parsed.hostname or "").lower()
        if parsed.scheme != "https" or not any(host == domain or host.endswith("." + domain) for domain in permitted):
            raise ValueError(f"Download host not on public-source allowlist: {host}")
        with SESSION.get(url, timeout=(10, 45), stream=True, allow_redirects=False) as response:
            if response.is_redirect:
                from urllib.parse import urljoin
                url = urljoin(url, response.headers["Location"])
                continue
            response.raise_for_status()
            payload = bytearray()
            for block in response.iter_content(65536):
                payload.extend(block)
                if len(payload) > 25_000_000:
                    raise ValueError("PDF exceeds 25 MB download limit")
            if not payload.startswith(b"%PDF"):
                raise ValueError("Source did not return a PDF")
            return bytes(payload), url
    raise ValueError("Too many PDF redirects")


def extract_pdf(payload):
    reader = PdfReader(io.BytesIO(payload))
    if reader.is_encrypted:
        raise ValueError("Encrypted PDF skipped; no access restrictions are bypassed")
    if len(reader.pages) > 300:
        raise ValueError("PDF exceeds 300-page extraction limit")
    pages = []
    for number, page in enumerate(reader.pages, 1):
        text = clean(page.extract_text())
        control_ratio = sum(0x7F <= ord(char) <= 0x9F or char == "\ufffd" for char in text) / max(1, len(text))
        boilerplate = text.upper().startswith("DISCLAIMER") or "This report was prepared as an account of work sponsored" in text[:250]
        readable_words = re.findall(r"\b[A-Za-z]{3,25}\b", text)
        if text and not boilerplate and control_ratio < 0.005 and len(readable_words) >= 20:
            pages.append({"page": number, "text": text})
    if sum(len(page["text"]) for page in pages) < 500:
        raise ValueError("PDF contains insufficient extractable text (OCR not enabled)")
    return pages, len(reader.pages)


def collect():
    ROOT.mkdir(parents=True, exist_ok=True)
    seeds = json.loads(SEEDS.read_text())
    records, failures = [], []
    select = "id,doi,title,publication_year,authorships,abstract_inverted_index,referenced_works,primary_topic,locations,best_oa_location"
    jobs = [
        ("https://api.openalex.org/works", {
            "search": query, "per_page": 25, "select": select,
            "filter": "publication_year:1980-2026",
        }, normalize_openalex)
        for query in seeds["queries"]
    ]
    jobs += [
        ("https://api.openalex.org/works", {
            "search": "extreme ultraviolet lithography", "per_page": 25, "select": select,
            "filter": f"authorships.institutions.country_code:{country},publication_year:1980-2026",
        }, normalize_openalex) for country in ("jp", "nl", "us")
    ]
    jobs += [
        ("https://api.openalex.org/works/W2111621196", {}, normalize_openalex),
        ("https://www.osti.gov/api/v1/records", {"search": "extreme ultraviolet lithography", "rows": 80}, normalize_osti),
        ("https://www.osti.gov/api/v1/records", {"search": "extreme ultraviolet lithography", "rows": 40, "sort": "publication_date", "order": "asc"}, normalize_osti),
    ]
    for url, params, normalize in jobs:
        try:
            response = cached_json(url, params)
            works = response if isinstance(response, list) else response.get("results", [response] if "id" in response else [])
            if isinstance(response, dict) and response.get("error"):
                raise ValueError(str(response["error"]))
            cache_path = ROOT / "raw" / (digest(url + json.dumps(params, sort_keys=True)) + ".json")
            retrieved_at = datetime.fromtimestamp(cache_path.stat().st_mtime, timezone.utc).isoformat()
            normalized = [normalize(work) for work in works]
            for record in normalized:
                for source in record["sources"]:
                    source["retrieved_at"] = retrieved_at
            records.extend(normalized)
            log.info("Collected %s: %d records", params.get("search", url), len(works))
        except (requests.RequestException, ValueError, KeyError) as exc:
            failures.append({"stage": "metadata", "url": url, "params": params, "error": str(exc)})
            log.warning("Metadata request failed: %s", exc)
    records = merge_records([
        r for r in records if r["year"] and 1980 <= r["year"] <= 2026
        and r["record_type"] not in ("Patent", "Software", "Dataset")
    ])
    # Enrich DOI-linked repository records with structured affiliations and abstracts.
    repository_dois = sorted({
        r["doi"] for r in records if r["doi"] and any(s["provider"] == "OSTI" for s in r["sources"])
    })
    enrichment = []
    for start in range(0, len(repository_dois), 25):
        params = {
            "filter": "doi:" + "|".join(repository_dois[start:start + 25]),
            "select": select, "per_page": 25,
        }
        try:
            response = cached_json("https://api.openalex.org/works", params)
            cache_path = ROOT / "raw" / (digest("https://api.openalex.org/works" + json.dumps(params, sort_keys=True)) + ".json")
            retrieved_at = datetime.fromtimestamp(cache_path.stat().st_mtime, timezone.utc).isoformat()
            for work in response["results"]:
                record = normalize_openalex(work)
                record["sources"][0]["retrieved_at"] = retrieved_at
                enrichment.append(record)
        except (requests.RequestException, ValueError, KeyError) as exc:
            failures.append({"stage": "enrichment", "error": str(exc)})
            log.warning("Repository metadata enrichment failed: %s", exc)
    records = merge_records(enrichment + records)
    if not records:
        raise RuntimeError("No research metadata could be collected")
    # Preserve country and historical coverage before filling the bounded corpus.
    selected = {}
    for country in ("JP", "NL", "US"):
        group = [r for r in records if country in r["countries"]]
        group.sort(key=lambda r: (r["year"] or 9999))
        for record in group[:25]:
            selected[record["id"]] = record
    with_pdfs = [r for r in records if r["pdf_urls"]]
    with_pdfs.sort(key=lambda r: (not any(s["provider"] == "OSTI" for s in r["sources"]), r["year"] or 9999))
    for record in with_pdfs[:50] + records:
        if len(selected) >= seeds["target_records"]:
            break
        selected[record["id"]] = record
    records = list(selected.values())
    pdf_dir = ROOT / "pdfs"
    pdf_dir.mkdir(exist_ok=True)
    full_text_count = attempts = 0
    for record in sorted(records, key=lambda r: (not bool(r["pdf_urls"]), r["year"] or 9999)):
        record["retrieved_at"] = datetime.now(timezone.utc).isoformat()
        record["pages"] = []
        record["pdf_page_count"] = None
        if full_text_count < seeds["target_full_text"] and attempts < seeds["max_full_text_attempts"]:
            for url in sorted(record["pdf_urls"], key=lambda u: "osti.gov" not in u):
                attempts += 1
                try:
                    path = pdf_dir / (digest(url) + ".pdf")
                    if path.exists():
                        payload, final_url = path.read_bytes(), url
                    else:
                        payload, final_url = download_pdf(url)
                        path.write_bytes(payload)
                    record["pages"], record["pdf_page_count"] = extract_pdf(payload)
                    record["full_text_url"] = final_url
                    record["content_hash"] = hashlib.sha256(payload).hexdigest()
                    full_text_count += 1
                    log.info("Extracted [%d] %s (%d pages)", full_text_count, record["title"], len(record["pages"]))
                    break
                except (requests.RequestException, ValueError, OSError, PyPdfError) as exc:
                    failures.append({"stage": "full_text", "paper_id": record["id"], "url": url, "error": str(exc)})
                    log.warning("Full text unavailable: %s: %s", record["title"], exc)
                if attempts >= seeds["max_full_text_attempts"]:
                    break
        record["coverage"] = "full_text" if record["pages"] else ("abstract" if record["abstract"] else "metadata")
    write_json(ROOT / "corpus.json", records)
    write_json(ROOT / "corpus-manifest.json", {
        "queries": seeds["queries"],
        "records": [{
            key: record.get(key) for key in (
                "id", "title", "doi", "aliases", "sources", "coverage", "content_hash", "full_text_url",
            )
        } for record in records],
    })
    report = {
        "records": len(records), "full_text": full_text_count,
        "abstract_only": sum(r["coverage"] == "abstract" for r in records),
        "metadata_only": sum(r["coverage"] == "metadata" for r in records),
        "countries": {country: sum(country in r["countries"] for r in records) for country in ("JP", "US", "NL", "Unknown")},
        "full_text_attempts": attempts, "failures": failures,
    }
    write_json(ROOT / "collection-report.json", report)
    log.info("Collection complete: %s", {k: v for k, v in report.items() if k != "failures"})
    return report
