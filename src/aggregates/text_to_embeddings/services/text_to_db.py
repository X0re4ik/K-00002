from src.shared.types import TensorType
from src.aggregates.text_to_embeddings.dto.EmbeddingMetadataDTO import (
    EmbeddingMetadataDTO,
)
from .embedding import EmbeddingService, embedding_service_factory
from .text_to_embedding import TextToEmbeddingService, text_to_embedding_service_factory
from .embedding_to_db import EmbeddingToDBService, embedding_to_db_service_factory

from src.models.embedding import get_embedding_size


from src.shared.libs.tsid import create_tsid


class TextToDBService:

    def __init__(
        self,
        text_to_embedding_service: TextToEmbeddingService,
        embedding_to_db_service: EmbeddingToDBService,
    ) -> None:
        self._text_to_embedding_service = text_to_embedding_service
        self._embedding_to_db_service = embedding_to_db_service

    async def text_to_db(self, text: str):

        embedding = self._text_to_embedding_service.text_to_embedding(
            text,
        )

        embedding_list = self._tensor_to_list(
            embedding,
        )

        await self._embedding_to_db_service.embedding_to_db(
            EmbeddingMetadataDTO(
                id=create_tsid(),
                embedding=embedding_list,
                payload={
                    "id": create_tsid(),
                    "text": text,
                },
            ),
        )

    def _tensor_to_list(self, embedding: TensorType) -> list[float]:
        assert (
            embedding.shape[0] == get_embedding_size()
        ), f"embedding.shape[0] != get_embedding_size() ({embedding.shape[0]} != {get_embedding_size()})"
        return embedding.tolist()


def text_to_db_service_factory():
    return TextToDBService(
        text_to_embedding_service=text_to_embedding_service_factory(),
        embedding_to_db_service=embedding_to_db_service_factory(),
    )
