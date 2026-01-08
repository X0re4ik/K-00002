from dataclasses import dataclass

from torch.types import Tensor


@dataclass
class TokenizeDTO:
    input_ids: Tensor
    attention_mask: Tensor
