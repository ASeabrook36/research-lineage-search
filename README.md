# EUV Research Atlas

A local semantic research-discovery prototype focused on the development of extreme
ultraviolet lithography: Japanese imaging and optics research, US national-laboratory
programs, Dutch and European research, and ASML's industrialization effort.

**Actual Cosmos DB, actual collected research, and local embeddings.** No Azure
resources, paid APIs, accounts, user-supplied keys, hosted inference, or generated
answers. Initial setup downloads public data, model weights, and dependencies.

## Use the populated prototype

Open **http://localhost:8000**. Search by topic or question, filter by affiliation
country, institution, author, field/topic, or publication year, and inspect source
passages and paper connections. The **Explore the lineage** tab provides attributed
historical milestones and recorded paper-to-paper citations.

Example searches:

- `Early Japanese soft X-ray imaging using multilayer mirrors`
- `ASML EUCLIDES European EUV lithography program`
- `US national laboratories and EUV lithography engineering test systems`
- `Tin plasma light sources for EUV lithography`

The prototype targets approximately **150 research records and 20–40 accessible
full-text sources**, with a cap of **2,500 genuine passage embeddings**. Actual
counts appear in the application and `/api/status`; extraction failures and corpus
coverage are recorded in `.local/collection-report.json`. This is a deliberately
small, incomplete research collection, not a comprehensive or balanced history.

## Fresh setup

Requires Docker Desktop running Linux containers. The tested machine is an
Apple Silicon Mac with approximately 8 GB allocated to Docker. Node.js is needed
only for browser tests; the application itself runs entirely in Docker.

```sh
docker compose build app
docker compose up -d cosmos
docker compose run --rm app python -m research probe
docker compose run --rm app python -m research collect
docker compose run --rm app python -m research ingest
docker compose up -d
```

The probe verifies actual vector ordering and database-side filtering before
research ingestion. Collection can encounter unavailable PDFs, encrypted documents,
publisher restrictions, API rate limits, and image-only documents. These failures
are reported rather than bypassed. A missing corpus or model produces an explicit
setup error, not fabricated results.

Model preparation occurs during ingestion. It downloads the pinned
`sentence-transformers/all-MiniLM-L6-v2` revision, saves it locally, and uses only
that local copy for queries. The image uses a CPU-only PyTorch build; no GPU is
required. The first build and downloads require appreciable disk space.

```sh
# Service readiness and logs
docker compose ps
curl http://localhost:8000/api/status
docker compose logs --tail=50 app cosmos
```

All published ports bind to `127.0.0.1`: web gateway `8000`, Cosmos gateway `8081`,
emulator readiness `8080`, and emulator Data Explorer `1234`. The container-network
gateway name is `cosmos`; the application is the supported presentation layer.
The emulator does not enforce authentication. Its SDK placeholder is not a
credential or Azure API key. Do not expose these services to untrusted networks.
Emulator and model-library telemetry are disabled.

## Offline use and persistence

After collection, ingestion, and model preparation, no network access is needed
for search, paper details, filters, or lineage navigation. Opening an external
publisher/source link still requires internet access.

For network-isolated operation, recreate the app and database on an internal
Docker network. A small loopback-published Nginx gateway bridges the browser to
that network; it can proxy only to the local app, not arbitrary external URLs.
This avoids Docker Desktop's internal-network published-port limitation and
preserves the named Cosmos volume and local caches:

```sh
docker compose down
docker compose -f compose.yaml -f compose.offline.yaml up -d
```

In this mode, use the application at port 8000. Direct host access to emulator
ports may be unavailable on Docker Desktop; its internal API remains available
to the app and to `docker compose exec` commands.

Return to the normal network before fetching new data:

```sh
docker compose -f compose.yaml -f compose.offline.yaml down
docker compose up -d
```

Use `docker compose stop` to stop services while retaining containers and data;
`docker compose start` restarts them. `docker compose down` removes containers,
not the named database volume. Restarting Docker or the services does not require
collecting the papers again.

Persistent local files are ignored by Git:

| Location | Contents |
| --- | --- |
| Docker volume `euv-research_cosmos-data` | Research records, passages/vectors, citation/history edges, ingestion configuration |
| `.local/raw/` | Cached public metadata API responses |
| `.local/pdfs/` | Accessible source PDFs for local processing only |
| `.local/models/` | Pinned embedding model and download cache |
| `.local/corpus.json` | Normalized corpus with extracted page text |
| `.local/corpus-manifest.json` | Collected source identifiers, URLs, coverage, and content hashes |
| `.local/embedding-cache.json` | Versioned embedding cache for repeat ingestion |
| `.local/*-report.json` | Collection, ingestion, and relevance reports |

To change the vector schema or embedding model, stop the app, explicitly delete
the prototype database, then rerun the probe and ingestion. This command destroys
**only this prototype database**, not unrelated Cosmos data:

```sh
docker compose stop app
docker compose run --rm app python -c \
  'from research.db import client, DATABASE; client().delete_database(DATABASE)'
docker compose run --rm app python -m research probe
docker compose run --rm app python -m research ingest
docker compose up -d
```

Ingestion is single-writer, restartable, and idempotent. It reconciles a paged
projection of document identities and synchronization hashes, upserts only changed
documents, and removes stale records. An unchanged repeat run performs no corpus
document writes. This intentionally retains an O(N) identity reconciliation to
detect missing or stale document identities without trusting a potentially stale
local manifest. Synchronization hashes are application-owned; manual payload
changes that leave a hash unchanged are not detected.
Source PDF `content_hash` values are preserved separately from internal
`sync_hash` values. Searches refuse incomplete or incompatible ingestion state
rather than mixing model versions. Recollection
uses the cached source responses; changing seed queries produces new requests.
Upstream search results can change, so retain the local manifest and raw cache
when reproducing a specific collection.

## What the evidence means

**Search results** are nearest-neighbor matches, grouped by paper, from Cosmos
`VectorDistance` queries. They are not answers generated by a language model.
The tested emulator exposes cosine similarity scores, with higher values meaning
more similar. Scores are not probabilities, historical importance, or confidence.
No application-side similarity engine replaces Cosmos.

**Recorded citations** point from a citing paper to the work it references, as
reported by OpenAlex. References outside the collected corpus are explicitly
labeled. Citation metadata is incomplete; a citation alone does not prove transfer
of ideas, collaboration, or causation.

**Curated historical connections** are attributed to
[ASML's 2022 retrospective](https://www.asml.com/en/company/stories/2022/making-euv-lab-to-fab).
They distinguish chronology from documented program connections. This company
account is useful context, not an exhaustive independent history. In particular,
the view does **not** claim that every US contribution descended directly from a
Japanese paper. German, British, Belgian, and other contributions are not excluded.

**Related research** in a paper's detail view is semantic similarity to a
representative indexed passage, never an inferred citation or historical influence.

Countries are institutional affiliations on the publication, not nationalities or
countries of invention. Papers can have multiple countries. Unknown metadata stays
unknown. Within a filter category, multiple choices use OR; different categories
use AND. Hold Command/Ctrl to select multiple options and apply the filters.

The initial version retrieves English text, including English abstracts of
Japanese research. There is no OCR or Japanese-language retrieval. Boilerplate-only,
encrypted, and visibly corrupted PDF extractions are rejected. Abstract-only and
metadata-only records are labeled; metadata-only records have no invented
searchable passages. A full-text label means a PDF yielded usable text, **not**
that every page was extractable or indexed. A fair round-robin passage cap can
omit later sections of long sources; the detail view exposes extracted/indexed
passage counts.

## Data sources and rights

[OpenAlex](https://help.openalex.org/api/) supplies bibliographic metadata,
abstracts where available, authors, institutions, affiliation countries, topics,
and references. It supports limited credential-free access; this prototype uses
no keys and never creates an account. Requests share its public IP rate budget.
Metadata is enriched by DOI when repository records have less structured data.

[OSTI](https://www.osti.gov/api/v1/docs) supplies DOE research metadata and
accessible technical reports, manuscripts, and proceedings. Public institutional
and publisher PDFs are fetched only from an explicit source-host allowlist.
Collection uses bounded retries, timeouts, a 25 MB/PDF size limit, a 300-page
extraction limit, and a bounded number of download attempts.

Accessibility does not imply permission to redistribute a PDF. Preserve source
links and supplied license information; PDFs and extracted full text remain in
the local ignored cache, not the repository. Consult each source's terms.
The embedding model is distributed under Apache-2.0; see its
[model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2).
Direct dependencies are declared in `requirements.txt`; `requirements.lock`
captures the resolved CPU runtime and is applied during image builds.

## Cosmos design and limitations

The pinned ARM-compatible vNext emulator supports the NoSQL API in gateway mode
and DiskANN vector indexing. It does **not** support full-text, flat-vector, or
quantized-flat indexes. This app therefore uses semantic vector search, not a
pretend full-text/hybrid search service.

- Passages use `/paper_id` as partition key, collocating a paper's passages.
  Search across this small research collection intentionally crosses partitions.
- `/embedding` uses **384 dimensions, float32, cosine, `diskANN`**. Vector elements
  are excluded from normal scalar indexing. Filters are applied inside Cosmos,
  before nearest-neighbor selection. Results are grouped and progressively
  over-fetched to avoid one long paper filling the result list.
- Research records and settings use `/id`. Citation/history edges use `/from_id`;
  an `incoming-edges` container uses `/to_id` and mirrors only citations to papers
  in the corpus. Paper connections query outgoing and incoming adjacency within
  their respective partitions, deduplicating self-citations. Both views are
  reconciled before ingestion is marked complete.
- Known settings documents use point reads, and paper-scoped passage queries
  explicitly supply the partition key. Cross-paper semantic search remains
  intentionally cross-partition.
- The first bounded vector query includes projected evidence, avoiding additional
  queries on the normal small-corpus path. If paper diversity requires progressive
  over-fetching, expanded queries return only passage ID, paper ID, and score;
  selected evidence is then hydrated with partition-scoped projections (at most
  two passages per returned paper). These avoid transferring stored embedding
  arrays. Over-fetching is bounded by the published passage count and logs total
  candidates across all attempts plus evidence-query counts. The expanded path
  trades additional scoped queries for smaller payloads; benchmark total RU and
  latency on Azure before claiming a cloud cost improvement.
- Scalar indexing excludes return-only payloads such as passage text, titles,
  source URLs, embedding elements, and synchronization hashes, while retaining
  filter fields. The emulator requires the include-all root policy; selective
  include-only paths were rejected by its index engine. Explicit exclusions are
  therefore used rather than a cloud-only include-only policy.
- A settings container records model/chunk/storage versions, materialized filter
  options, published corpus counts, and ingestion completion. Filter options are
  generation-checked against the completed ingestion. Status/health checks read
  the settings document instead of counting containers. Use
  `/api/status?live_counts=true` for on-demand database counts; `counts_source`
  distinguishes live counts from completed-ingestion counts. Out-of-band database
  edits require re-ingestion to refresh materialized metadata.
- Query logs include elapsed time and accumulated response charges across pages;
  vector-search logs include all over-fetch attempts and evidence queries. SDK
  throttle retry counts and wait time are recorded when available. Missing RU
  headers are reported as unknown, not zero. The SDK retains its built-in bounded,
  retry-after-aware throttling policy; ingestion writes remain serial.
- Windows contain at most 200 content tokens with 32-token overlap and a title
  prefix of at most 40 tokens. Actual model tokenization is checked against its
  256-token limit; input is not silently truncated.
- The tested emulator omits `vectorEmbeddingPolicy` from container reads even
  though it retains the DiskANN index definition. The capability probe verifies
  known-vector behavior, while a settings record preserves the requested policy.
- Cloud documentation describes DiskANN training requirements and scanning below
  1,000 vectors. This prototype does not pad the corpus or claim index acceleration
  from a successful policy write. It does not establish cloud RU charges,
  throughput, production recall, or cloud/emulator performance parity.
- Range/composite index optimization and policy updates are limited in the emulator.
  Recreate the database for schema/model changes rather than relying on a
  successful but potentially no-op collection update. Initialization reads back
  every container's partition and scalar/vector index definitions and rejects
  mismatches; `create_container_if_not_exists` does not migrate existing policies.

### Storage migration without recollecting

The `/kind` edge layout from the initial prototype is incompatible with storage
version 2. Preserve the old database volume and use a separate Compose project
to migrate safely. Stop the previous project's services first (without `down -v`),
copy its ignored `.local/` directory into this worktree if needed, and verify that
`corpus.json`, `embedding-cache.json`, PDFs, and `models/minilm/` are present.
Do **not** run `collect` for a baseline-preserving migration.

The following uses the cached corpus and model on an internal network, with a
separate database volume named `euv-research-kit_cosmos-data`:

```sh
docker compose -p euv-research-kit -f compose.yaml -f compose.offline.yaml build app
docker compose -p euv-research-kit -f compose.yaml -f compose.offline.yaml up -d cosmos
docker compose -p euv-research-kit -f compose.yaml -f compose.offline.yaml run --rm app python -m research probe
docker compose -p euv-research-kit -f compose.yaml -f compose.offline.yaml run --rm \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 app python -m research ingest
docker compose -p euv-research-kit -f compose.yaml -f compose.offline.yaml up -d
```

Retain the same `-p` and `-f` arguments for subsequent commands against this stack.
The app is still available at **http://localhost:8000**. With a complete baseline
embedding cache, ingestion reports zero new embeddings; PDFs and model files are
neither fetched nor modified. Model or cache omissions must be resolved from the
baseline, not by recollecting data. Application image dependencies may need a
networked build when their Docker layers are not already cached.

References: [emulator feature support](https://learn.microsoft.com/en-us/azure/cosmos-db/emulator-linux),
[vector policy and indexing](https://learn.microsoft.com/en-us/azure/cosmos-db/vector-search).

## Verification

```sh
# Unit tests (no downloads; use the built app image)
docker compose run --rm app python -m pytest -q -m 'not integration'

# Real Cosmos integration: populated data and cached model required
docker compose run --rm -e RUN_INTEGRATION=1 app python -m pytest -q

# Reviewed topic relevance set and measured latency
docker compose run --rm app python -m research evaluate

# Browser tests against the running app, including source/lineage navigation
npm ci
npx playwright install chromium
npm run test:ui
```

With the offline stack running, explicitly check blocked backend egress as well:

```sh
docker compose exec -T -e RUN_INTEGRATION=1 -e REQUIRE_OFFLINE=1 app python -m pytest -q
```

The evaluation set is in `data/evaluation.json`. It measures whether an expected
relevant topic/paper appears in the top five, with a target of 80%; this is not a
historical-influence or exhaustive recall benchmark. Inspect the report's actual
retrieved titles and misses. Compact general-purpose embeddings can rank generic
EUV overviews above a more specific US-program paper; filtering and a more precise
query can help. No result is presented as proof of a complete research genealogy.

The populated development snapshot contains 150 records, 30 sources yielding usable
PDF text, 111 abstract-only records, 9 metadata-only records, 2,500 indexed passages,
138 in-collection citation links, and 7 attributed historical connections.
The reviewed 10-question set retrieved an expected relevant result in the top five
for all 10 questions after repository-metadata enrichment. These small-corpus
observations are not guarantees for new questions or future upstream data.

Browser tests block remote page requests and check search, country filters, paper
details, lineage evidence, invalid dates, empty results, and mobile width. Running
them with `compose.offline.yaml` also exercises the backend on an internal network.
Integration tests compare a small filtered Cosmos ranking to an exact cosine
calculation used **only as a test oracle**.

## Retrieval API and future grounded answers

FastAPI serves the UI and structured endpoints:

| Endpoint | Purpose |
| --- | --- |
| `GET /api/search?q=...` | Semantic retrieval; repeated `country`, `field`, `institution`, `author`, plus `year_from`, `year_to`, `limit` |
| `GET /api/filters` | Corpus-derived filter identifiers and display labels |
| `GET /api/papers/{id}` | Metadata and indexed passages with page/source provenance |
| `GET /api/papers/{id}/connections` | Recorded citations and separately labeled semantic neighbors |
| `GET /api/lineage` | Attributed milestones, historical connections, and in-corpus citations |
| `GET /api/status` | Published ingestion counts, readiness, and corpus reports; `?live_counts=true` requests current database counts |

Stable paper/passage IDs, source URLs, page references, and structured retrieval
results are the integration boundary for a future local grounded-answer layer.
That layer should cite these passages and preserve evidence labels; it is
deliberately not implemented in this prototype.
