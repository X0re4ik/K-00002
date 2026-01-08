from dataclasses import dataclass
from torch.types import Tensor

@dataclass
class Embedding1DDTO:
    embedding: Tensor