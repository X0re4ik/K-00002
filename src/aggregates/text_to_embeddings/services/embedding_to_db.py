from dataclasses import dataclass, field
from src.aggregates.text_to_embeddings.dto.EmbeddingMetadataDTO import EmbeddingMetadataDTO


from src.shared.api import api_qdrant_client


from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from src.shared.configs import PROJECT_SETTINGS


from typing import TypedDict


class EmbeddingToDBService:

    def __init__(
        self,
        qdrant_client: QdrantClient,
        collection_name: str,
    ) -> None:
        self._qdrant_client = qdrant_client
        self._collection_name = collection_name

    async def embedding_to_db(
        self,
        dto: EmbeddingMetadataDTO,
    ) -> None:
        self._qdrant_client.upload_points(
            collection_name=self._collection_name,
            points=[
                PointStruct(
                    id=dto.id,
                    vector=dto.embedding,
                    payload=dto.payload,
                ),
            ],
            batch_size=64,
            parallel=1,
            max_retries=5,
            wait=False,
        )


def embedding_to_db_service_factory():
    return EmbeddingToDBService(
        qdrant_client=api_qdrant_client,
        collection_name=PROJECT_SETTINGS.api.qdrant.text_collection,
    )
