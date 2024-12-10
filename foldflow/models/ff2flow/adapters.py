from typing import Dict, Tuple

import torch
from esm.esmfold.v1.trunk import RelativePosition
from torch import nn

IMPLEMENTED_REPRESENTATION = ["bb", "seq"]


class ProjectConcatRepresentation(nn.Module):
    def __init__(
        self,
        input_dims_single: Dict[str, int],
        input_dims_pair: Dict[str, int],
        single_dim: int,
        pair_dim: int,
        layer_norm: bool = True,
    ):
        super().__init__()
        self.single_dim = single_dim
        self.pair_dim = pair_dim
        self.modalities_name = list(input_dims_single.keys())
        assert all((k in IMPLEMENTED_REPRESENTATION for k in self.modalities_name))

        # Initialize projections
        self.projections_single = nn.ModuleDict(
            {
                k: nn.Linear(input_dims_single[k], single_dim)
                for k in self.modalities_name
            }
        )
        self.projections_pair = nn.ModuleDict(
            {k: nn.Linear(input_dims_pair[k], pair_dim) for k in self.modalities_name}
        )

        # Layer normalization
        self.layer_norm = layer_norm
        if layer_norm:
            self.layer_norm_single = nn.LayerNorm(
                len(self.modalities_name) * single_dim
            )
            self.layer_norm_pair = nn.LayerNorm(len(self.modalities_name) * pair_dim)

    @property
    def out_single_dim(self):
        return self.single_dim * len(self.modalities_name)

    @property
    def out_pair_dim(self):
        return self.pair_dim * len(self.modalities_name)

    def forward(
        self, single: Dict[str, torch.Tensor], pair: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # Project and concatenate
        single_repr = torch.cat(
            [self.projections_single[k](single[k]) for k in self.modalities_name],
            dim=-1,
        )
        pair_repr = torch.cat(
            [self.projections_pair[k](pair[k]) for k in self.modalities_name], dim=-1
        )

        # Apply layer normalization if enabled
        if self.layer_norm:
            single_repr = self.layer_norm_single(single_repr)
            pair_repr = self.layer_norm_pair(pair_repr)

        return single_repr, pair_repr


class TrunkToDecoderNetwork(nn.Module):
    def __init__(self, single_in: int, pair_in: int, single_out: int, pair_out: int):
        super().__init__()
        self.lin_single = nn.Linear(single_in, single_out)
        self.lin_pair = nn.Linear(pair_in, pair_out)
        self.single_layernorm = nn.LayerNorm(single_out)
        self.pair_layernorm = nn.LayerNorm(pair_out)

    def forward(self, single, pair):
        single_out = self.single_layernorm(self.lin_single(single))
        pair_out = self.pair_layernorm(self.lin_pair(pair))
        return single_out, pair_out


class SequenceToTrunkNetwork(nn.Module):
    def __init__(
        self,
        esm_single_dim: int,
        num_layers: int,
        d_single: int,
        esm_attn_dim: int,
        d_pair: int,
        position_bins: int,
        pairwise_state_dim: int,
    ):
        super().__init__()
        self.pair_mlp = nn.Sequential(
            nn.LayerNorm(esm_attn_dim),
            nn.Linear(esm_attn_dim, d_pair),
            nn.ReLU(),
            nn.Linear(d_pair, d_pair),
        )
        self.single_mlp = nn.Sequential(
            nn.LayerNorm(esm_single_dim),
            nn.Linear(esm_single_dim, d_single),
            nn.ReLU(),
            nn.Linear(d_single, d_single),
        )
        self.esm_single_combine = nn.Parameter(torch.zeros(num_layers + 1))

        self.pairwise_positional_embedding = RelativePosition(
            position_bins, pairwise_state_dim
        )

    def forward(self, seq_emb_s, seq_emb_z, res_idx, res_mask):
        # single rpr
        seq_emb_s = seq_emb_s.to(self.esm_single_combine.dtype)
        seq_emb_s = seq_emb_s.detach()
        seq_emb_s = (
            self.esm_single_combine.softmax(0).unsqueeze(0) @ seq_emb_s
        ).squeeze(2)
        single = self.single_mlp(seq_emb_s)

        # pair rpr
        seq_emb_z = seq_emb_z.to(self.esm_single_combine.dtype)
        seq_emb_z = seq_emb_z.detach()
        pair = self.pair_mlp(seq_emb_z)
        pair = pair + self.pairwise_positional_embedding(res_idx, mask=res_mask)
        return single, pair


