import dataclasses
from typing import Optional, Tuple

import esm
import torch
import tree
from esm.data import Alphabet
from openfold.np import residue_constants
from torch import nn

load_fn = esm.pretrained.load_model_and_alphabet
ESM_REGISTRY = {
    "esm2_8M_270K": esm.pretrained.esm2_t6_8M_UR50D,
    "esm2_35M_270K": esm.pretrained.esm2_t12_35M_UR50D,
    "esm2_650M": esm.pretrained.esm2_t33_650M_UR50D,
    "esm2_3B": esm.pretrained.esm2_t36_3B_UR50D,
    "esm2_15B": esm.pretrained.esm2_t48_15B_UR50D,
}

SINGLE_REPNS_SEQ_LEN_SHAPE_IDX = 1


@dataclasses.dataclass
class SequenceData:
    aa_sequence: torch.Tensor
    non_linker_mask: torch.Tensor

    @classmethod
    def from_single_chain(cls, aa_sequence):
        return cls(aa_sequence, torch.ones_like(aa_sequence, dtype=torch.bool))


class FrozenEsmModel(nn.Module):
    def __init__(
        self,
        model_key: str,
        use_esm_attn_map: bool = True,
    ):
        super().__init__()
        self.esm, self.esm_dict = ESM_REGISTRY[model_key]()
        self.register_buffer("af2_to_esm", FrozenEsmModel._af2_to_esm(self.esm_dict))
        self.use_esm_attn_map = use_esm_attn_map
        self._repr_layers = tuple(range(self.esm.num_layers + 1))
        self._previous_call = None

    @property
    def repr_layers(self):
        return self._repr_layers

    @property
    def single_dim(self):
        return self.esm.embed_dim

    @property
    def attn_head(self):
        return self.esm.attention_heads

    @property
    def num_layers(self):
        return self.esm.num_layers

    @torch.no_grad()
    def forward(
        self,
        aa_sequence: torch.Tensor,
        chain_idx: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        seq_mask: Optional[torch.Tensor] = None,
        convert_to_esm: bool = True,
        cache_last_call: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Get seq_len before the sequence is augmented
        seq_len = aa_sequence.shape[1]

        # preprocesss
        if convert_to_esm:
            # TODO: need to mask sequence after adding linkers ?
            aa_sequence = self._convert_af_to_esm(aa_sequence, attn_mask, seq_mask)

        sequence_data = self._add_special_tokens(aa_sequence, chain_idx)

        # run
        if (
            cache_last_call
            and self._previous_call is not None
            and self._previous_call["inputs"].shape == sequence_data.aa_sequence.shape
            and torch.all(self._previous_call["inputs"] == sequence_data.aa_sequence)
        ):
            return self._previous_call["outputs"]
        res = self.esm(
            sequence_data.aa_sequence,
            repr_layers=self.repr_layers,
            need_head_weights=self.use_esm_attn_map,
        )

        # postprocess
        single_repns = torch.stack(
            [v for _, v in sorted(res["representations"].items())], dim=2
        )
        single_repns = single_repns[:, 1:-1]  # B, L, nLayers, C
        pair_repns = (
            res["attentions"].permute(0, 4, 3, 1, 2).flatten(3, 4)[:, 1:-1, 1:-1, :]
            if self.use_esm_attn_map
            else None
        )

        if cache_last_call:
            self._previous_call = {
                "inputs": sequence_data.aa_sequence.clone().detach(),
                "outputs": tree.map_structure(
                    lambda x: x.clone().detach(), (single_repns, pair_repns)
                ),
            }

        return single_repns, pair_repns

    @staticmethod
    def _af2_to_esm(d: Alphabet):
        # Remember that t is shifted from residue_constants by 1 (0 is padding).
        esm_reorder = [d.padding_idx] + [
            d.get_idx(v) for v in residue_constants.restypes_with_x
        ]
        return torch.tensor(esm_reorder)

    def _convert_af_to_esm(self, aa_sequence, attn_mask, seq_mask):
        """attn_mask is passed to the model for masking attention/batchiung
        seq_mask is the masking pattern for masking residues, e.g. during LLM training
        """
        aa_sequence = self._af2_idx_to_esm_idx(aa_sequence, attn_mask)
        if seq_mask is not None:
            aa_sequence = self._mask_inputs_to_esm(aa_sequence, seq_mask)
        return aa_sequence

    def _af2_idx_to_esm_idx(self, aa, mask):
        aa = (aa + 1).masked_fill(mask != 1, 0)
        self.af2_to_esm = self.af2_to_esm.to(
            aa.device
        )
        return self.af2_to_esm[aa]

    def _mask_inputs_to_esm(self, esmaa, pattern):
        new_esmaa = esmaa.clone()
        new_esmaa[pattern == 1] = self.esm_dict.mask_idx
        return new_esmaa

    def _add_special_tokens(
        self, aa_sequence: torch.Tensor, chain_idx: torch.Tensor
    ) -> SequenceData:
        batch_size = aa_sequence.size(0)
        """Adds bos/eos/linker tokens for the language model, since the structure module doesn't use these."""
        sequence_data = SequenceData.from_single_chain(aa_sequence)

        bosi, eosi = self.esm_dict.cls_idx, self.esm_dict.eos_idx
        bos = sequence_data.aa_sequence.new_full((batch_size, 1), bosi)
        eos = sequence_data.aa_sequence.new_full(
            (batch_size, 1), self.esm_dict.padding_idx
        )
        sequence_data.aa_sequence = torch.cat(
            [bos, sequence_data.aa_sequence, eos], dim=1
        )
        # Use the first padding index as eos during inference.
        sequence_data.aa_sequence[
            range(batch_size), (sequence_data.aa_sequence != 1).sum(1)
        ] = eosi
        # return aa_sequence
        return sequence_data
