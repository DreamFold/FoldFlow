# FF modular model for designs with multiple modalities.

from foldflow.models.ff2flow.adapters import (
    ProjectConcatRepresentation,
    SequenceToTrunkNetwork,
    TrunkToDecoderNetwork,
)
from foldflow.models.ff2flow.structure_network import (
    EmbedderConfig,
    FF2StructureNetwork,
    FF2StructureNetworkConfig,
)
from foldflow.models.ff2flow.trunk import (
    FF2TrunkBlockConfig,
    FF2TrunkTransformer,
)
from foldflow.models.components.sequence.frozen_esm import FrozenEsmModel
from functools import lru_cache
from foldflow.models.se3_fm import SE3FlowMatcher

dependency = lambda fn : property(lru_cache()(fn))

class FF2Dependencies:

    def __init__(self, config):
        self.config = config
            
    @dependency
    def flow_matcher(self):
        return SE3FlowMatcher(self.config.flow_matcher)

    @dependency
    def bb_encoder(self) -> FF2StructureNetwork:
        emb_conf = EmbedderConfig(
            use_alphafold_position_embedding=self.config.model.embed.use_alphafold_position_embedding,
            embed_self_conditioning=self.config.model.embed.embed_self_conditioning,
            relpos_k=self.config.model.embed.relpos_k,
        )
        self.bb_encoder_conf = FF2StructureNetworkConfig(
            num_blocks=self.config.model.bb_encoder.num_blocks,  # FFOT uses 4
            coordinate_scaling=self.config.model.bb_encoder.coordinate_scaling,
            do_last_edge_update=True,
            embed=emb_conf,
        )
        bb_encoder = FF2StructureNetwork(
            self.bb_encoder_conf,
            flow_matcher=self.flow_matcher,
            generate_sc_angles=False,
        )
        return bb_encoder

    @dependency
    def bb_decoder(self) -> FF2StructureNetwork:
        emb_conf = EmbedderConfig(
            use_alphafold_position_embedding=self.config.model.embed.use_alphafold_position_embedding,
            embed_self_conditioning=self.config.model.embed.embed_self_conditioning,
            relpos_k=self.config.model.embed.relpos_k,
        )
        self.bb_decoder_conf = FF2StructureNetworkConfig(
            num_blocks=self.config.model.bb_decoder.num_blocks,
            coordinate_scaling=self.config.model.bb_decoder.coordinate_scaling,
            do_last_edge_update=False,
            embed=emb_conf,
        )
        bb_decoder = FF2StructureNetwork(
            self.bb_decoder_conf,
            flow_matcher=self.flow_matcher,
            generate_sc_angles=False,
        )
        return bb_decoder

    @dependency
    def seq_encoder(self):
        # load & set up the model
        esm_wrapper = FrozenEsmModel(
            self.config.model.esm2_model_key,
        )
        esm_wrapper.esm.eval()
        esm_wrapper.esm.requires_grad_(False)
        esm_wrapper.esm.half()

        return esm_wrapper

    @dependency
    def sequence_to_trunk_network(self):
        _block_config = FF2TrunkBlockConfig(
            sequence_state_dim=self.combiner_network.out_single_dim,
            pairwise_state_dim=self.combiner_network.out_pair_dim,
            sequence_head_width=self.config.model.modalities_transformer.sequence_head_width,
            pairwise_head_width=self.config.model.modalities_transformer.pairwise_head_width,
            chunk_size=self.config.model.modalities_transformer.chunk_size,
        )

        pair_dim = (
            self.seq_encoder.num_layers * self.seq_encoder.attn_head
        )  
        sequence_to_trunk_network = SequenceToTrunkNetwork(
            esm_single_dim=self.seq_encoder.single_dim,
            num_layers=self.seq_encoder.num_layers,
            d_single=self.config.model.seq_emb_to_block.single_dim,
            esm_attn_dim=pair_dim,
            d_pair=self.config.model.seq_emb_to_block.pair_dim,
            position_bins=_block_config.position_bins,
            pairwise_state_dim=self.config.model.seq_emb_to_block.pair_dim,
        )
        return sequence_to_trunk_network

    @dependency
    def combiner_network(self):
        single_dim = self.config.model.representation_combiner.single_dim
        pair_dim = self.config.model.representation_combiner.pair_dim
        single_dims = {
            "bb": self.bb_encoder_conf.ipa.c_s,
            "seq": self.config.model.seq_emb_to_block.single_dim,
        }
        pair_dims = {
            "bb": self.bb_encoder_conf.ipa.c_z,
            "seq": self.config.model.seq_emb_to_block.pair_dim,
        }
        self.rpr_comb_single_dim = single_dim
        self.rpr_comb_pair_dim = pair_dim
        combiner_network = ProjectConcatRepresentation(
            input_dims_single=single_dims,
            input_dims_pair=pair_dims,
            single_dim=single_dim,
            pair_dim=pair_dim,
            layer_norm=self.config.model.representation_combiner.layer_norm,
        )
        return combiner_network

    @dependency
    def trunk_network(self):
        trunk_network = self._get_trunk_transformer()
        return trunk_network

    def _get_trunk_transformer(self):
        block_config = FF2TrunkBlockConfig(
            sequence_state_dim=self.combiner_network.out_single_dim,
            pairwise_state_dim=self.combiner_network.out_pair_dim,
            sequence_head_width=self.config.model.modalities_transformer.sequence_head_width,
            pairwise_head_width=self.config.model.modalities_transformer.pairwise_head_width,
            chunk_size=self.config.model.modalities_transformer.chunk_size,
        )
        return FF2TrunkTransformer(
            num_blocks=self.config.model.modalities_transformer.num_blocks,
            block_config=block_config,
        )

    @dependency
    def trunk_to_decoder_network(self):
        trunk_to_decoder_network = TrunkToDecoderNetwork(
            single_in=self.combiner_network.out_single_dim,
            single_out=self.bb_decoder_conf.ipa.c_s,
            pair_in=self.combiner_network.out_pair_dim,
            pair_out=self.bb_decoder_conf.ipa.c_z,
        )
        return trunk_to_decoder_network

    @property
    def time_embedder(self):
        return None
