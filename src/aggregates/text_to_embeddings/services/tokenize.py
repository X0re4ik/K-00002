from transformers import AutoTokenizer, AutoModel

from src.aggregates.text_to_embeddings.dto.TokenizeDTO import TokenizeDTO


class TokenizeService:

    def __init__(self, tokenizer_model) -> None:
        self._tokenizer_model = tokenizer_model

    def tokenize(self, text: str) -> TokenizeDTO:

        batch_dict = self._tokenizer_model(
            text,
            max_length=512,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        return TokenizeDTO(
            input_ids=batch_dict["input_ids"],
            attention_mask=batch_dict["attention_mask"],
        )


def tokenize_service_factory():
    return TokenizeService(
        tokenizer_model=AutoTokenizer.from_pretrained(
            "sentence-transformers/distiluse-base-multilingual-cased-v2",
        ),
    )
