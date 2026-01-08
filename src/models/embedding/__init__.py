from transformers import AutoModel


EMBEDDING_MODEL = AutoModel.from_pretrained(
    "sentence-transformers/distiluse-base-multilingual-cased-v2",
)


def get_embedding_size() -> int:
    return EMBEDDING_MODEL.config.hidden_size
