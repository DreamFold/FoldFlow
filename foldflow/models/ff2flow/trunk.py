from typing import Optional, Tuple

import torch
from esm.esmfold.v1.tri_self_attn_block import TriangularSelfAttentionBlock
from pydantic.dataclasses import dataclass
from torch import nn


@dataclass
class FF2TrunkBlockConfig:
    sequence_state_dim: int = 1024  # ESMF default
    pairwise_state_dim: int = 128  # ESMF default
    sequence_head_width: int = 32  # ESMF default
    pairwise_head_width: int = 32  # ESMF default
    dropout: float = 0.0  # ESMF default
    position_bins: int = 32  # ESMF default
    chunk_size: Optional[int] = None


class FF2TrunkTransformer(nn.Module):
    def __init__(
        self,
        num_blocks: int = 1,
        block_config: FF2TrunkBlockConfig = FF2TrunkBlockConfig(),
    ):
        super().__init__()

        assert block_config.sequence_state_dim % block_config.sequence_head_width == 0
        assert block_config.pairwise_state_dim % block_config.pairwise_head_width == 0

        self.blocks = nn.ModuleList(
            [
                TriangularSelfAttentionBlock(
                    sequence_state_dim=block_config.sequence_state_dim,
                    pairwise_state_dim=block_config.pairwise_state_dim,
                    sequence_head_width=block_config.sequence_head_width,
                    pairwise_head_width=block_config.pairwise_head_width,
                    dropout=block_config.dropout,
                )
                for _ in range(num_blocks)
            ]
        )
        self.chunk_size = block_config.chunk_size

    def forward(
        self,
        single: torch.Tensor,
        pair: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        for block in self.blocks:
            single, pair = block(single, pair, mask=mask, chunk_size=self.chunk_size)
        return single, pair
