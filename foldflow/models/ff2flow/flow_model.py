from typing import Dict, Tuple

import torch
from foldflow.data import all_atom
from foldflow.models.ff2flow.adapters import (
    ProjectConcatRepresentation,
    SequenceToTrunkNetwork,
    TrunkToDecoderNetwork,
)
from foldflow.models.ff2flow.ff2_dependencies import FF2Dependencies
from foldflow.models.ff2flow.structure_network import FF2StructureNetwork
from foldflow.models.ff2flow.trunk import FF2TrunkTransformer
from foldflow.models.components.sequence.frozen_esm import FrozenEsmModel
from openfold.utils import rigid_utils as ru
from torch import nn
from foldflow.models.components.sequence.frozen_esm import ESM_REGISTRY
from foldflow.models.se3_fm import SE3FlowMatcher


class FF2Model(nn.Module):
    def __init__(
        self,
        config,
        flow_matcher: SE3FlowMatcher,
        bb_encoder: FF2StructureNetwork,
        bb_decoder: FF2StructureNetwork,
        seq_encoder: FrozenEsmModel,
        sequence_to_trunk_network: SequenceToTrunkNetwork,
        combiner_network: ProjectConcatRepresentation,
        trunk_network: FF2TrunkTransformer,
        trunk_to_decoder_network: TrunkToDecoderNetwork,
        time_embedder,
    ):
        super().__init__()
        self.config = config
        self.flow_matcher = flow_matcher
        self.bb_encoder = bb_encoder
        self.bb_decoder = bb_decoder
        self.seq_encoder = seq_encoder
        self.sequence_to_trunk_network = sequence_to_trunk_network
        self.combiner_network = combiner_network
        self.trunk_network = trunk_network
        self.trunk_to_decoder_network = trunk_to_decoder_network
        self.time_embedder = time_embedder

        self._is_conditional_generation = False
        self._is_scaffolding_generation = False

    @property
    def is_cond_seq(self):
        return self.config.is_cond_seq

    @classmethod
    def from_dependencies(cls, dependencies: FF2Dependencies):
        return cls(
            dependencies.config,
            dependencies.flow_matcher,
            dependencies.bb_encoder,
            dependencies.bb_decoder,
            dependencies.seq_encoder,
            dependencies.sequence_to_trunk_network,
            dependencies.combiner_network,
            dependencies.trunk_network,
            dependencies.trunk_to_decoder_network,
            dependencies.time_embedder,
        )
    @classmethod
    def from_ckpt(cls, ckpt: Dict[str, torch.Tensor], deps: FF2Dependencies):
        _prefix_to_remove = "vectorfield_network."
        ckpt["state_dict"] = {
            k.replace(_prefix_to_remove, ""): v for k, v in ckpt["state_dict"].items()
        }
        model = cls.from_dependencies(deps)
        ckpt_lm_name = ckpt["esm_model"]
        assert (
            deps.config.model.esm2_model_key == ckpt_lm_name
        ), f"Model trained with different ESM2 {ckpt_lm_name}, but got {deps.config.model.esm2_model_key=}"
        cls._add_esm_to_ckpt(model, ckpt)
        model.load_state_dict(ckpt["state_dict"])
        return model
    
    @staticmethod
    def _add_esm_to_ckpt(model, ckpt: Dict[str, torch.Tensor]) -> None:
        for k, v in model.seq_encoder.state_dict().items():
            if k.startswith("esm."):
                ckpt["state_dict"][f"seq_encoder.{k}"] = v

    def _get_vectorfields(
        self,
        pred_rigids: torch.Tensor,
        init_rigids: torch.Tensor,
        t: torch.Tensor,
        res_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        _ ,rot_vectorfield = self.flow_matcher.calc_rot_vectorfield(
            pred_rigids.get_rots().get_rot_mats(),
            init_rigids.get_rots().get_rot_mats(),
            t,
        )
        rot_vectorfield = rot_vectorfield * res_mask[..., None, None]
        trans_vectorfield = self.flow_matcher.calc_trans_vectorfield(
            pred_rigids.get_trans(),
            init_rigids.get_trans(),
            t[:, None, None],
            scale=True,
        )
        trans_vectorfield = trans_vectorfield * res_mask[..., None]
        return rot_vectorfield, trans_vectorfield

    def _make_seq_mask_pattern(self, batch):
        aatype = batch["aatype"]
        if self.is_scaffolding_generation:
            return 1.0 - batch["fixed_mask_seq"]
        if self._is_conditional_generation:
            # no masking during conditional generation
            return torch.zeros_like(aatype)

        if not self.training:
            # mask the entire sequence for inference
            return torch.ones_like(aatype)

        pattern = torch.zeros_like(aatype)
        rows_to_mask = (
            torch.rand(aatype.shape[0]) < self.config.model.p_mask_sequence
        ).to(aatype.device)
        pattern[rows_to_mask] = 1
        return pattern

    @property
    def is_conditional_generation(self) -> bool:
        return self._is_conditional_generation

    def conditional_generation(self):
        self.eval()
        self._is_conditional_generation = True

    @property
    def is_scaffolding_generation(self) -> bool:
        return self._is_scaffolding_generation

    def scaffolding_generation(self):
        self._is_scaffolding_generation = True

    def train(self, is_training):
        # nn.Module.eval() calls self.train(False)
        self._is_conditional_generation = False
        super().train(is_training)

    def forward(
        self, batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:  # TODO: verify the return type.
        device = batch["rigids_t"].device
        init_rigids = ru.Rigid.from_tensor_7(batch["rigids_t"])
        t = batch["t"]

        bb_mask = batch["res_mask"].type(torch.float32)  # [B, N]
        edge_mask = bb_mask[..., None] * bb_mask[..., None, :]

        # Sequence representations.
        seq_mask_pattern = self._make_seq_mask_pattern(batch)

        seq_emb_s, seq_emb_z = self.seq_encoder(
            batch["aatype"],
            batch["chain_idx"],
            attn_mask=batch["res_mask"],
            seq_mask=seq_mask_pattern,
        )
        seq_emb_s, seq_emb_z = seq_emb_s.to(device), seq_emb_z.to(device)
        # Processing of the sequence emb (trainable). # LN and Lin. layers.
        seq_emb_s, seq_emb_z = self.sequence_to_trunk_network(
            seq_emb_s, seq_emb_z, batch["seq_idx"], batch["res_mask"]
        )

        # Structure representations.
        bb_encoder_output = self.bb_encoder(
            res_mask=batch["res_mask"],
            fixed_mask=batch["fixed_mask"],
            seq_idx=batch["seq_idx"],
            chain_idx=batch["chain_idx"],
            t=t,
            rigids_t=batch["rigids_t"],
            unscale_rigids=False,
            self_conditioning_ca=batch["sc_ca_t"],
        )
        bb_emb_s = bb_encoder_output["single_emb"]
        bb_emb_z = bb_encoder_output["pair_emb"]
        rigids_updated = bb_encoder_output["rigids"]
        init_single_embed = bb_encoder_output["init_single_embed"]
        init_pair_embed = bb_encoder_output["init_pair_embed"]

        # Representations combiner
        single_representation = {"bb": bb_emb_s, "seq": seq_emb_s}
        pair_representation = {"bb": bb_emb_z, "seq": seq_emb_z}
        single_embed, pair_embed = self.combiner_network(
            single_representation, pair_representation
        )

        # Evoformer or linear or identity.
        single_embed, pair_embed = self.trunk_network(
            single_embed, pair_embed, mask=batch["res_mask"].float()
        )

        # Update representations dim for decoder.
        single_embed, pair_embed = self.trunk_to_decoder_network(
            single_embed, pair_embed
        )

        single_embed = 0.5 * (single_embed + init_single_embed)
        pair_embed = 0.5 * (pair_embed + init_pair_embed)

        # edge and node masking
        single_embed = single_embed * bb_mask[..., None]
        pair_embed = pair_embed * edge_mask[..., None]

        # update the rigids with the new single and pair representation.
        bb_decoder_output = self.bb_decoder(
            res_mask=batch["res_mask"],
            fixed_mask=batch["fixed_mask"],
            t=t,
            single_embed=single_embed,
            pair_embed=pair_embed,
            rigids_t=rigids_updated.to_tensor_7(),
            unscale_rigids=False,
        )
        rigids_updated = bb_decoder_output["rigids"]
        psi = bb_decoder_output["psi"]
        if self._is_scaffolding_generation:
            mask = batch["fixed_mask"][:, :, None]
            gt_psi = batch["torsion_angles_sin_cos"][..., 2, :]
            psi = psi * (1 - mask) + gt_psi * mask

        res_mask = batch["res_mask"].type(torch.float32)
        rot_vectorfield, trans_vectorfield = self._get_vectorfields(
            rigids_updated, init_rigids, t, res_mask
        )
        model_out: Dict[str, torch.Tensor] = {}
        model_out["rot_vectorfield"] = rot_vectorfield
        model_out["trans_vectorfield"] = trans_vectorfield
        model_out["psi"] = psi
        model_out["rigids"] = rigids_updated.to_tensor_7()
        bb_representations = all_atom.compute_backbone(rigids_updated, psi)
        model_out["atom37"] = bb_representations[0].to(device)
        model_out["atom14"] = bb_representations[-1].to(device)

        return model_out
