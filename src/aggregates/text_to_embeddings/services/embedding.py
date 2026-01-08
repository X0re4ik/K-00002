from torch.types import Tensor
from transformers import AutoTokenizer, AutoModel
from torch.nn.functional import normalize


from src.aggregates.text_to_embeddings.dto.Embedding1DDTO import Embedding1DDTO
from src.aggregates.text_to_embeddings.dto.TokenizeDTO import TokenizeDTO


class EmbeddingService:

    def __init__(self, embedding_model) -> None:
        self._embedding_model = embedding_model

    def to_embedding(self, dto: TokenizeDTO) -> Embedding1DDTO:

        batch_dict__ = {
            "input_ids": dto.input_ids,
            "attention_mask": dto.attention_mask,
        }
        output = self._embedding_model(**batch_dict__)
        embedding = self._average_pool(
            output.last_hidden_state,
            dto.attention_mask,
        )
        embedding = normalize(
            embedding,
            p=2,
            dim=1,
        )
        embedding = embedding.cpu().detach().squeeze()

        return Embedding1DDTO(
            embedding=embedding,
        )

    def _average_pool(self, last_hidden_states: Tensor, attention_mask: Tensor):
        last_hidden = last_hidden_states.masked_fill(
            ~attention_mask[..., None].bool(), 0.0
        )
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def embedding_service_factory():
    return EmbeddingService(
        embedding_model=AutoModel.from_pretrained(
            "sentence-transformers/distiluse-base-multilingual-cased-v2",
        ),
    )
