from dataclasses import dataclass, field




from qdrant_client.models import Payload as PointStructPayload




@dataclass
class EmbeddingMetadataDTO:
    id: int
    embedding: list[float]
    payload: PointStructPayload | None = field(
        default=None,
    )