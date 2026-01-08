from .embedding import embedding_service_factory, EmbeddingService
from .tokenize import tokenize_service_factory, TokenizeService


from torch.types import Tensor


class TextToEmbeddingService:

    def __init__(
        self,
        embedding_service: EmbeddingService,
        tokenize_service: TokenizeService,
    ) -> None:
        self._embedding_service = embedding_service
        self._tokenize_service = tokenize_service

    def text_to_embedding(self, text: str) -> Tensor:
        token_dto = self._tokenize_service.tokenize(
            text,
        )
        embedding_1d_dto = self._embedding_service.to_embedding(
            token_dto,
        )
        return embedding_1d_dto.embedding


def text_to_embedding_service_factory():
    return TextToEmbeddingService(
        embedding_service=embedding_service_factory(),
        tokenize_service=tokenize_service_factory(),
    )
