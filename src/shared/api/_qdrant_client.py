from qdrant_client import models, QdrantClient
from src.shared.configs import PROJECT_SETTINGS

api_qdrant_client = QdrantClient(
    url=PROJECT_SETTINGS.api.qdrant.http_url,
    check_compatibility=False,
)


def create_collection(
    qdrant_client: QdrantClient,
    collection_name: str,
    vector_size: int,
):
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=vector_size,
            distance=models.Distance.COSINE,
        ),
    )


def create_collect_if_not_exists(
    qdrant_client: QdrantClient,
    collection_name: str,
    vector_size: int,
):
    _collections_res = qdrant_client.get_collections()
    if collection_name in [collection.name for collection in _collections_res.collections]:
        return
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=vector_size,
            distance=models.Distance.COSINE,
        ),
    )
