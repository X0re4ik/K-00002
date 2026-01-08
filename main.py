from src.aggregates.text_to_embeddings.services import text_to_embedding_service_factory


t = text_to_embedding_service_factory()


print(t.text_to_embedding("I LOve IUCH"))
