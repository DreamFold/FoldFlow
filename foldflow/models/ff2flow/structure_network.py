from typing import Optional

import torch
from foldflow.models.components.ipa_pytorch import IpaNetwork
from foldflow.models.components.network import Embedder
from openfold.model.structure_module import AngleResnet
from pydantic.dataclasses import dataclass

# Corresponds to the "benchmark" configuration of our
# base FF model without the backbone update.


@dataclass
class IPAConfig:
    c_z: int
    c_hidden: int = 256
    c_skip: int = 64
    no_heads: int = 8
    no_qk_points: int = 8
    no_v_points: int = 12
    seq_tfmr_num_heads: int = 4
    seq_tfmr_num_layers: int = 2
    num_blocks: int = 4
    coordinate_scaling: float = 1.0  # No scaling by default.
    context_embed_size: Optional[int] = None
    context_mid_embed_size: int = 256
    context_embed_init_size: int = 2560
    attn_type: str = "cross"
    c_s: int = 256
    update_bb: bool = False
    do_last_edge_update: bool = False


@dataclass
class EmbedderConfig:
    index_embed_size: int = 32
    aatype_embed_size: int = 64
    split_chains: bool = False
    embed_self_conditioning: bool = True
    num_bins: int = 22
    min_bin: float = 1e-5
    max_bin: float = 20.0
    relpos_k: Optional[int] = None
    use_alphafold_position_embedding: Optional[bool] = False


@dataclass
class FF2StructureNetworkConfig:
    node_embed_size: int = 256
    edge_embed_size: int = 128
    c_hidden: int = 256
    c_skip: int = 64
    no_heads: int = 8
    no_qk_points: int = 8
    no_v_points: int = 12
    seq_tfmr_num_heads: int = 4
    seq_tfmr_num_layers: int = 2
    num_blocks: int = 4
    coordinate_scaling: float = 0.1
    context_embed_size: int = None
    context_mid_embed_size: int = 256
    context_embed_init_size: int = 2560
    attn_type: str = "cross"
    use_context: bool = False
    embed: EmbedderConfig = EmbedderConfig()
    do_last_edge_update: bool = False
    ipa: Optional[IPAConfig] = None

    def __post_init__(self):
        self.ipa = IPAConfig(
            c_z=self.edge_embed_size,
            c_s=self.node_embed_size,
            no_heads=self.no_heads,
            no_qk_points=self.no_qk_points,
            no_v_points=self.no_v_points,
            seq_tfmr_num_heads=self.seq_tfmr_num_heads,
            seq_tfmr_num_layers=self.seq_tfmr_num_layers,
            num_blocks=self.num_blocks,
            coordinate_scaling=self.coordinate_scaling,
            context_embed_size=self.context_embed_size,
            context_mid_embed_size=self.context_mid_embed_size,
            context_embed_init_size=self.context_embed_init_size,
            attn_type=self.attn_type,
            do_last_edge_update=self.do_last_edge_update,
        )


class FF2StructureNetwork(IpaNetwork):
    def __init__(self, model_conf, flow_matcher, generate_sc_angles=False):
        super().__init__(model_conf, flow_matcher)
        self.embedding_layer = Embedder(model_conf)
        self.generate_sc_angles = generate_sc_angles

    def _get_node_edge_embed(
        self,
        res_mask: torch.Tensor,
        fixed_mask: torch.Tensor,
        seq_idx: torch.Tensor,
        t: torch.Tensor,
        self_conditioning_ca,
    ):

        # Frames as [batch, res, 7] tensors.
        bb_mask = res_mask.type(torch.float32)  # [B, N]
        fixed_mask = fixed_mask.type(torch.float32)
        edge_mask = bb_mask[..., None] * bb_mask[..., None, :]

        # Initial embeddings of positonal and relative indices.
        init_node_embed, init_edge_embed = self.embedding_layer(
            seq_idx=seq_idx,
            t=t,
            fixed_mask=fixed_mask,
            self_conditioning_ca=self_conditioning_ca,
        )
        edge_embed = init_edge_embed * edge_mask[..., None]
        node_embed = init_node_embed * bb_mask[..., None]

        return node_embed, edge_embed

    def forward(
        self,
        res_mask: torch.Tensor,
        fixed_mask: torch.Tensor,
        t: torch.Tensor,
        rigids_t: Optional[torch.Tensor],
        single_embed: Optional[torch.Tensor] = None,
        pair_embed: Optional[torch.Tensor] = None,
        seq_idx: Optional[torch.Tensor] = None,
        chain_idx: Optional[torch.Tensor] = None,
        unscale_rigids: bool = False,
        aatype: Optional[torch.Tensor] = None,
        self_conditioning_ca: Optional[torch.Tensor] = None,
    ):
        assert (single_embed is None) == (pair_embed is None)
        if single_embed is None and pair_embed is None:
            single_embed, pair_embed = self._get_node_edge_embed(
                res_mask=res_mask,
                fixed_mask=fixed_mask,
                seq_idx=seq_idx,
                t=t,
                self_conditioning_ca=self_conditioning_ca,
            )
        init_single_embed = single_embed.clone()
        init_pair_embed = pair_embed.clone()
        input_feats = {}
        input_feats["res_mask"] = res_mask
        input_feats["fixed_mask"] = fixed_mask
        input_feats["rigids_t"] = rigids_t
        input_feats["t"] = t
        model_out = super().forward(
            init_node_embed=single_embed,
            edge_embed=pair_embed,
            input_feats=input_feats,
            encoder_mode=True,
        )
        single_emb = model_out["node_embed"]
        pair_emb = model_out["edge_embed"]
        rigids = (
            model_out["final_rigids"]
            if not unscale_rigids
            else model_out["scaled_rigids"]
        )
        psi = model_out["psi"]
        output_dict = {
            "single_emb": single_emb,
            "pair_emb": pair_emb,
            "rigids": rigids,
            "psi": psi,
            "init_single_embed": init_single_embed,
            "init_pair_embed": init_pair_embed,
        }
        return output_dict
