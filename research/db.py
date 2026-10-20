import os
from functools import lru_cache
from urllib.parse import urlparse

from azure.cosmos import CosmosClient, PartitionKey

DATABASE = "euv-research"
DIMENSIONS = 384
VECTOR_POLICY = {
    "vectorEmbeddings": [{
        "path": "/embedding", "dataType": "float32",
        "distanceFunction": "cosine", "dimensions": DIMENSIONS,
    }]
}
INDEX_POLICY = {
    "indexingMode": "consistent",
    "includedPaths": [{"path": "/*"}],
    "excludedPaths": [{"path": "/embedding/*"}],
    "vectorIndexes": [{"path": "/embedding", "type": "diskANN"}],
}


@lru_cache(maxsize=1)
def client():
    endpoint = os.environ.get("COSMOS_ENDPOINT", "http://localhost:8081")
    # vNext does not enforce authentication; the SDK requires a syntactic key.
    parsed = urlparse(endpoint)
    if parsed.scheme != "http" or parsed.hostname not in {"localhost", "127.0.0.1", "cosmos"} or parsed.username or parsed.password:
        raise ValueError("Only the local Cosmos emulator is supported")
    return CosmosClient(endpoint, credential="bG9jYWwtZW11bGF0b3I=", enable_endpoint_discovery=False)


def connect(initialize=False):
    db = client().create_database_if_not_exists(DATABASE) if initialize else client().get_database_client(DATABASE)
    if initialize:
        db.create_container_if_not_exists("papers", partition_key=PartitionKey("/id"))
        db.create_container_if_not_exists("edges", partition_key=PartitionKey("/kind"))
        db.create_container_if_not_exists("settings", partition_key=PartitionKey("/id"))
        db.create_container_if_not_exists(
            "passages", partition_key=PartitionKey("/paper_id"),
            vector_embedding_policy=VECTOR_POLICY, indexing_policy=INDEX_POLICY,
        )
    return db


def query(container, sql, parameters=None):
    return list(container.query_items(
        query=sql, parameters=parameters or [], enable_cross_partition_query=True,
    ))


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
