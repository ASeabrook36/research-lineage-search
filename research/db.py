import logging
import os
import time
from dataclasses import dataclass
from functools import lru_cache
from urllib.parse import urlparse

from azure.core.paging import ItemPaged
from azure.cosmos import CosmosClient, PartitionKey
from azure.cosmos.exceptions import CosmosHttpResponseError, CosmosResourceNotFoundError

log = logging.getLogger(__name__)
DATABASE = "euv-research"
STORAGE_VERSION = 2
DIMENSIONS = 384
VECTOR_POLICY = {
    "vectorEmbeddings": [{
        "path": "/embedding", "dataType": "float32",
        "distanceFunction": "cosine", "dimensions": DIMENSIONS,
    }]
}


def index_policy(*excluded_paths):
    return {
        "indexingMode": "consistent",
        "includedPaths": [{"path": "/*"}],
        "excludedPaths": [{"path": path} for path in excluded_paths],
    }


INDEX_POLICY = {
    **index_policy(
        "/embedding/*", "/text/?", "/title/?", "/source_url/?", "/coverage/?",
        "/page/?", "/section/?", "/chunk_version/?", "/sync_hash/?",
    ),
    "vectorIndexes": [{"path": "/embedding", "type": "diskANN"}],
}
CONTAINERS = {
    "papers": ("id", index_policy(
        "/abstract/?", "/aliases/*", "/authors/*", "/institutions/*", "/sources/*",
        "/references/*", "/pdf_urls/*", "/licenses/*", "/title/?", "/sync_hash/?",
    )),
    "edges": ("from_id", index_policy(
        "/from_title/?", "/to_title/?", "/evidence_url/?", "/explanation/?",
        "/relationship/?", "/year/?", "/to_id/?", "/sync_hash/?",
    )),
    "incoming-edges": ("to_id", index_policy(
        "/from_title/?", "/to_title/?", "/evidence_url/?", "/explanation/?",
        "/relationship/?", "/year/?", "/from_id/?", "/kind/?", "/target_in_corpus/?", "/sync_hash/?",
    )),
    "settings": ("id", index_policy("/options/*", "/report/*", "/vector_policy/*")),
    "passages": ("paper_id", INDEX_POLICY),
}


@dataclass
class RequestMetrics:
    responses: int = 0
    charged_responses: int = 0
    request_charge: float = 0.0
    throttle_retries: int = 0
    throttle_wait_ms: float = 0.0

    def record(self, headers, response):
        # query_items also calls the hook before fetching, with stale headers.
        if isinstance(response, ItemPaged):
            return
        self.responses += 1
        charge = headers.get("x-ms-request-charge")
        if charge is not None:
            self.charged_responses += 1
            self.request_charge += float(charge)
        self.throttle_retries += int(headers.get("x-ms-throttle-retry-count", 0))
        self.throttle_wait_ms += float(headers.get("x-ms-throttle-retry-wait-time-ms", 0))

    def summary(self):
        return {
            "responses": self.responses,
            "request_charge": round(self.request_charge, 3) if self.charged_responses else None,
            "charge_complete": self.responses > 0 and self.responses == self.charged_responses,
            "throttle_retries": self.throttle_retries,
            "throttle_wait_ms": self.throttle_wait_ms,
        }


@lru_cache(maxsize=1)
def client():
    endpoint = os.environ.get("COSMOS_ENDPOINT", "http://localhost:8081")
    # vNext does not enforce authentication; the SDK requires a syntactic key.
    parsed = urlparse(endpoint)
    if parsed.scheme != "http" or parsed.hostname not in {"localhost", "127.0.0.1", "cosmos"} or parsed.username or parsed.password:
        raise ValueError("Only the local Cosmos emulator is supported")
    return CosmosClient(
        endpoint, credential="bG9jYWwtZW11bGF0b3I=",
        enable_endpoint_discovery=False, connection_timeout=10,
    )


def validate_container(container, partition_field, indexing_policy, vector_policy=None):
    properties = container.read()
    if properties.get("partitionKey", {}).get("paths") != [f"/{partition_field}"]:
        raise RuntimeError(f"Incompatible partition key on {container.id}; use the documented storage migration")
    actual = properties.get("indexingPolicy", {})
    for name in ("includedPaths", "excludedPaths"):
        expected_paths = {p["path"] for p in indexing_policy[name]}
        actual_paths = {p["path"] for p in actual.get(name, [])}
        # Cosmos can append its automatic system-property exclusion.
        actual_paths -= {"/_etag/?", '/"_etag"/?'}
        if name == "includedPaths":
            actual_paths -= {"/_ts/?", '/"_ts"/?'}
        if expected_paths != actual_paths:
            raise RuntimeError(f"Incompatible {name} on {container.id}; use the documented storage migration")
    if actual.get("indexingMode") != indexing_policy["indexingMode"]:
        raise RuntimeError(f"Incompatible indexing mode on {container.id}")
    if actual.get("vectorIndexes", []) != indexing_policy.get("vectorIndexes", []):
        raise RuntimeError(f"Incompatible vector index on {container.id}")
    if vector_policy is not None:
        stored = properties.get("vectorEmbeddingPolicy")
        if stored is None:
            log.warning("Emulator omits vectorEmbeddingPolicy; run the known-vector capability probe")
        elif stored != vector_policy:
            raise RuntimeError(f"Incompatible Cosmos vector policy on {container.id}")


def connect(initialize=False):
    db = client().create_database_if_not_exists(DATABASE) if initialize else client().get_database_client(DATABASE)
    if initialize:
        for name, (partition_field, policy) in CONTAINERS.items():
            vector = {"vector_embedding_policy": VECTOR_POLICY} if name == "passages" else {}
            container = db.create_container_if_not_exists(
                name, partition_key=PartitionKey(f"/{partition_field}"),
                indexing_policy=policy, **vector,
            )
            validate_container(container, partition_field, policy, vector.get("vector_embedding_policy"))
    return db


def read_item(container, item_id, partition_key, *, metrics=None):
    metrics = metrics if metrics is not None else RequestMetrics()
    start = time.perf_counter()
    try:
        return container.read_item(item_id, partition_key=partition_key, response_hook=metrics.record)
    finally:
        log.info(
            "Cosmos point read container=%s elapsed_ms=%.1f metrics=%s",
            container.id, (time.perf_counter() - start) * 1000, metrics.summary(),
        )


def read_optional(container, item_id):
    try:
        return read_item(container, item_id, item_id)
    except CosmosResourceNotFoundError:
        return None


def iter_query(container, sql, parameters=None, *, partition_key=None, metrics=None):
    metrics = metrics if metrics is not None else RequestMetrics()
    start = time.perf_counter()
    rows = 0
    try:
        for item in container.query_items(
            query=sql, parameters=parameters or [],
            partition_key=partition_key, enable_cross_partition_query=partition_key is None,
            max_item_count=100, response_hook=metrics.record,
        ):
            rows += 1
            yield item
    except CosmosHttpResponseError as exc:
        log.error(
            "Cosmos query failed container=%s status=%s activity_id=%s retry_after_ms=%s",
            container.id, exc.status_code, exc.headers.get("x-ms-activity-id"),
            exc.headers.get("x-ms-retry-after-ms"),
        )
        raise
    finally:
        log.info(
            "Cosmos query container=%s scoped=%s rows=%d elapsed_ms=%.1f metrics=%s",
            container.id, partition_key is not None, rows,
            (time.perf_counter() - start) * 1000, metrics.summary(),
        )


def query(container, sql, parameters=None, *, partition_key=None, metrics=None):
    return list(iter_query(container, sql, parameters, partition_key=partition_key, metrics=metrics))


def probe():
    """Test actual vector operations, not just accepted policy metadata."""
    db = connect(True)
    container = db.create_container_if_not_exists(
        "capability-probe", partition_key=PartitionKey("/paper_id"),
        vector_embedding_policy=VECTOR_POLICY, indexing_policy=INDEX_POLICY,
    )
    fixtures = [
        ("near", [1.0, 0.0], "JP"),
        ("middle", [0.6, 0.8], "US"),
        ("far", [-1.0, 0.0], "NL"),
    ]
    for id_, vector, country in fixtures:
        container.upsert_item({
            "id": id_, "paper_id": id_, "embedding": vector + [0.0] * (DIMENSIONS - 2),
            "countries": [country],
        })
    parameters = [{"name": "@v", "value": [1.0] + [0.0] * (DIMENSIONS - 1)}]
    sql = "SELECT TOP 3 c.id, VectorDistance(c.embedding, @v) AS distance FROM c ORDER BY VectorDistance(c.embedding, @v)"
    rows = query(container, sql, parameters)
    if [row["id"] for row in rows] != ["near", "middle", "far"]:
        raise RuntimeError(f"Cosmos vector ordering failed: {rows}")
    filtered = query(container, sql.replace("FROM c ORDER", "FROM c WHERE ARRAY_CONTAINS(c.countries, 'US') ORDER"), parameters)
    if [row["id"] for row in filtered] != ["middle"]:
        raise RuntimeError(f"Cosmos filtered vector retrieval failed: {filtered}")
    persisted = db.get_container_client("capability-probe").read_item("near", partition_key="near")
    if persisted["countries"] != ["JP"]:
        raise RuntimeError("Cosmos read-after-write failed")
    return {"vector_order": rows, "filter": filtered, "read_after_write": True}
