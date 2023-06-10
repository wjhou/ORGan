from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from tokenizer import Tokenizer
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers.activations import ACT2FN
from transformers.file_utils import ModelOutput
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
)
from transformers.models.bart.configuration_bart import BartConfig
from transformers.models.bart.modeling_bart import BartPretrainedModel

from src.models.modeling_bart import *
from src.models.modeling_bart import _expand_mask, _make_causal_mask


@dataclass
class VLModelOutput(ModelOutput):
    visual_last_hidden_state: torch.FloatTensor = None
    tag_last_hidden_state: torch.FloatTensor = None
    visual_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    node_cls_logits: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class VLSeq2SeqModelOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_visual_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_visual_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_tag_last_hidden_state: torch.FloatTensor = None
    node_cls_logits: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class VLSeq2SeqLMOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_visual_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_visual_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None


class VLBartDecoder(BartPretrainedModel):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a [`BartDecoderLayer`]

    Args:
        config: BartConfig
        embed_tokens (nn.Embedding): output embedding
    """

    def __init__(
        self,
        config: BartConfig,
        embed_tokens: Optional[nn.Embedding] = None,
    ):
        super().__init__(config)
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0

        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = nn.Embedding(
                config.vocab_size,
                config.d_model,
                self.padding_idx,
            )

        self.embed_positions = BartLearnedPositionalEmbedding(
            config.max_position_embeddings,
            config.d_model,
        )
        layers = [VLBartDecoderLayer(config) for _ in range(config.decoder_layers)]
        if config.outline_level > 0:
            layers = layers + [TDReasonLayer(config)]
        self.layers = nn.ModuleList(layers)
        self.layernorm_embedding = nn.LayerNorm(config.d_model)

        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def _prepare_decoder_attention_mask(
        self, attention_mask, input_shape, inputs_embeds, past_key_values_length
    ):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                past_key_values_length=past_key_values_length,
            ).to(self.device)
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(
                attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]
            )
            combined_attention_mask = (
                expanded_attn_mask
                if combined_attention_mask is None
                else expanded_attn_mask + combined_attention_mask,
            )
        return combined_attention_mask

    def forward(
        self,
        input_ids,
        attention_mask=None,
        encoder_visual_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_visual_attention_mask: Optional[torch.LongTensor] = None,
        encoder_node_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_node_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError(
                "You have to specify either decoder_input_ids or decoder_inputs_embeds"
            )

        # past_key_values_length
        past_key_values_length = (
            past_key_values[0][0].shape[2] if past_key_values is not None else 0
        )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, input_shape, inputs_embeds, past_key_values_length
        )
        # expand encoder attention mask
        if (
            encoder_visual_hidden_states is not None
            and encoder_visual_attention_mask is not None
        ):
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            encoder_visual_attention_mask = _expand_mask(
                encoder_visual_attention_mask,
                inputs_embeds.dtype,
                tgt_len=input_shape[-1],
            )

        # embed positions
        positions = self.embed_positions(input_shape, past_key_values_length)
        hidden_states = inputs_embeds + positions
        hidden_states = self.layernorm_embedding(hidden_states)

        hidden_states = nn.functional.dropout(
            hidden_states,
            p=self.dropout,
            training=self.training,
        )
        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = (
            ()
            if (output_attentions and encoder_visual_hidden_states is not None)
            else None
        )
        next_decoder_cache = () if use_cache else None

        # check if head_mask/cross_attn_head_mask has a correct number of layers specified if desired
        for attn_mask, mask_name in zip(
            [head_mask, cross_attn_head_mask], ["head_mask", "cross_attn_head_mask"]
        ):
            if attn_mask is not None:
                if attn_mask.size()[0] != (len(self.layers)):
                    raise ValueError(
                        "The `{mask_name}` should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."
                    )

        for idx, decoder_layer in enumerate(self.layers):
            # if idx == len(self.layers) - 1:
            #
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):
                continue

            past_key_value = (
                past_key_values[idx] if past_key_values is not None else None
            )

            if self.gradient_checkpointing and self.training:
                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, output_attentions, use_cache)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    encoder_visual_hidden_states,
                    encoder_visual_attention_mask,
                    encoder_node_hidden_states,
                    encoder_node_attention_mask,
                    head_mask[idx] if head_mask is not None else None,
                    cross_attn_head_mask[idx]
                    if cross_attn_head_mask is not None
                    else None,
                    None,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    encoder_visual_hidden_states=encoder_visual_hidden_states,
                    encoder_visual_attention_mask=encoder_visual_attention_mask,
                    encoder_node_hidden_states=encoder_node_hidden_states,
                    encoder_node_attention_mask=encoder_node_attention_mask,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    cross_attn_layer_head_mask=(
                        cross_attn_head_mask[idx]
                        if cross_attn_head_mask is not None
                        else None
                    ),
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[3 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

                if encoder_visual_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_cache,
                    all_hidden_states,
                    all_self_attns,
                    all_cross_attentions,
                ]
                if v is not None
            )

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        )


class TDReasonLayer(nn.Module):
    def __init__(self, config: BartConfig):
        super().__init__()
        self.config = config
        self.dropout = config.dropout
        self.embed_dim = config.d_model
        self.self_attn = BartAttention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
        )
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.encoder_tdr_attn = BartAttention(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
        )
        self.encoder_tdr_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_visual_hidden_states: Optional[torch.Tensor] = None,
        encoder_visual_attention_mask: Optional[torch.Tensor] = None,
        encoder_node_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_node_attention_mask: Optional[torch.LongTensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        cross_attn_layer_head_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = True,
    ) -> Tuple[
        torch.FloatTensor,
        Optional[
            Tuple[
                torch.FloatTensor,
                torch.FloatTensor,
            ]
        ],
    ]:
        residual = hidden_states

        # Self Attention
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = (
            past_key_value[:2] if past_key_value is not None else None
        )
        # add present self-attn cache to positions 1,2 of present_key_value tuple
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=self_attn_past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        hidden_states = nn.functional.dropout(
            hidden_states,
            p=self.dropout,
            training=self.training,
        )
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        hidden_states, cross_attn_weights, cross_attn_present_key_value = self.trr(
            hidden_states=hidden_states,
            encoder_node_hidden_states=encoder_node_hidden_states,
            encoder_node_attention_mask=encoder_node_attention_mask,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
        )

        present_key_value = present_key_value + cross_attn_present_key_value

        # Fully Connected
        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.activation_dropout, training=self.training
        )
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.dropout, training=self.training
        )
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)

        if use_cache:
            outputs += (present_key_value,)

        return outputs

    def trr(
        self,
        hidden_states: torch.Tensor,
        encoder_node_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_node_attention_mask: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
    ):
        tdr_cross_attn_past_key_value = (
            past_key_value[-2:] if past_key_value is not None else None
        )
        tgt_len = hidden_states.size(1)

        # 1st->2nd->3rd
        for i in range(self.config.outline_level):
            residual = hidden_states
            encoder_attention_mask_ = (encoder_node_attention_mask == i + 1).float()
            encoder_attention_mask = _expand_mask(
                encoder_attention_mask_,
                hidden_states.dtype,
                tgt_len=tgt_len,
            )
            (
                hidden_states,
                cross_attn_weights,
                cross_attn_present_key_value,
            ) = self.encoder_tdr_attn(
                hidden_states=hidden_states,
                key_value_states=encoder_node_hidden_states,
                attention_mask=encoder_attention_mask,
                past_key_value=tdr_cross_attn_past_key_value,
                output_attentions=output_attentions,
            )
            hidden_states = nn.functional.dropout(
                hidden_states,
                p=self.dropout,
                training=self.training,
            )
            gate = (
                encoder_attention_mask_.sum(dim=-1, keepdim=True).unsqueeze(-1) > 0
            ).float()
            hidden_states = (
                gate * self.encoder_tdr_attn_layer_norm(residual + hidden_states)
                + (1.0 - gate) * residual
            )
        return hidden_states, cross_attn_weights, cross_attn_present_key_value


class VLBartDecoderLayer(nn.Module):
    def __init__(self, config: BartConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.d_model

        self.self_attn = BartAttention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
        )
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.encoder_obs_attn = BartAttention(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
        )
        self.encoder_obs_attn_layer_norm = nn.LayerNorm(self.embed_dim)

        self.encoder_visual_attn = BartAttention(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
        )
        self.encoder_visual_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_visual_hidden_states: Optional[torch.Tensor] = None,
        encoder_visual_attention_mask: Optional[torch.Tensor] = None,
        encoder_node_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_node_attention_mask: Optional[torch.LongTensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        cross_attn_layer_head_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = True,
    ) -> Tuple[
        torch.FloatTensor,
        Optional[
            Tuple[
                torch.FloatTensor,
                torch.FloatTensor,
            ]
        ],
    ]:
        residual = hidden_states

        # Self Attention
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = (
            past_key_value[:2] if past_key_value is not None else None
        )
        # add present self-attn cache to positions 1,2 of present_key_value tuple
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=self_attn_past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.dropout, training=self.training
        )
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # Cross-Attention Block for Visual Representations
        start = 2

        # tdr_cross_attn_present_key_value = None
        if encoder_node_hidden_states is not None:
            residual = hidden_states
            tdr_cross_attn_past_key_value = (
                past_key_value[start : start + 2]
                if past_key_value is not None
                else None
            )
            # only attend the 1st-level feature (i.e., Observation Representations)
            encoder_attention_mask_ = (encoder_node_attention_mask == 1).float()
            encoder_attention_mask = _expand_mask(
                encoder_attention_mask_,
                hidden_states.dtype,
                tgt_len=hidden_states.size(1),
            )
            (
                hidden_states,
                cross_attn_weights,
                tdr_cross_attn_present_key_value,
            ) = self.encoder_obs_attn(
                hidden_states=hidden_states,
                key_value_states=encoder_node_hidden_states,
                past_key_value=tdr_cross_attn_past_key_value,
                attention_mask=encoder_attention_mask,
                layer_head_mask=layer_head_mask,
                output_attentions=output_attentions,
            )
            hidden_states = nn.functional.dropout(
                hidden_states,
                p=self.dropout,
                training=self.training,
            )
            hidden_states = self.encoder_obs_attn_layer_norm(residual + hidden_states)
            present_key_value = present_key_value + tdr_cross_attn_present_key_value
            start += 2

        # Cross-Attention Block for Observation Representations
        cross_attn_weights = None
        if encoder_visual_hidden_states is not None:
            residual = hidden_states
            visual_cross_attn_past_key_value = (
                past_key_value[start : start + 2]
                if past_key_value is not None
                else None
            )
            (
                hidden_states,
                cross_attn_weights,
                visual_cross_attn_present_key_value,
            ) = self.encoder_visual_attn(
                hidden_states=hidden_states,
                key_value_states=encoder_visual_hidden_states,
                attention_mask=encoder_visual_attention_mask,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=visual_cross_attn_past_key_value,
                output_attentions=output_attentions,
            )
            hidden_states = nn.functional.dropout(
                hidden_states,
                p=self.dropout,
                training=self.training,
            )
            hidden_states = self.encoder_visual_attn_layer_norm(
                residual + hidden_states
            )

            present_key_value = present_key_value + visual_cross_attn_present_key_value
            start += 2

        # Fully Connected
        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.activation_dropout, training=self.training
        )
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.dropout, training=self.training
        )
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class VLBartEncoder(BartPretrainedModel):
    def __init__(
        self,
        config: BartConfig,
        visual_encoder: VisualBartEncoder,
        shared: nn.Embedding,
    ):
        super().__init__(config)
        self.visual_encoder = visual_encoder
        self.tagset_size = config.tag_size
        self.dropout = config.dropout
        self.shared = shared
        self.rgcn = GraphBartEncoder(config, shared)
        self.cls_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, 1),
        )

    def forward(
        self,
        input_pixels=None,
        input_ids=None,
        attention_mask=None,
        obs_labels=None,
        node_ids=None,
        node_mask=None,
        matrix=None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, VLModelOutput]:
        node_ids = node_ids.masked_fill(
            node_ids == -100,
            self.rgcn.node_embed.padding_idx,
        )
        node_hidden_states = None
        node_attention_mask = (matrix != 0).float()
        node_hidden_states = self.rgcn(
            node_ids=node_ids,
            node_mask=node_mask,
            attention_mask=node_attention_mask,
        ).last_hidden_state

        visual_encoder_outputs = self.visual_encoder(
            input_pixels=input_pixels,
            outline_ids=node_hidden_states,
            outline_mask=(node_mask > 0).float(),
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        tmp_node_mask = None
        node_logits = None
        hidden_states = visual_encoder_outputs.last_hidden_state
        aligned_node_hidden_states = hidden_states[:, -node_hidden_states.size(1) :]
        visual_last_hidden_state = hidden_states[:, : -node_hidden_states.size(1)]
        node_logits = self.cls_head(aligned_node_hidden_states).squeeze(-1)
        aligned_node_hidden_states = node_hidden_states
        if obs_labels is not None:
            tmp_node_mask = obs_labels
        else:
            tmp_node_mask = (node_logits > 0).float()
        return VLModelOutput(
            visual_last_hidden_state=visual_last_hidden_state,
            tag_last_hidden_state=aligned_node_hidden_states,
            attentions=tmp_node_mask,
            node_cls_logits=node_logits,
        )


class BartModel(BartPretrainedModel):
    def __init__(self, config: BartConfig, visual_encoder):
        super().__init__(config)
        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.visual_embed = VisualEmbedding(config)
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)
        self.visual_encoder = VisualBartEncoder(
            config,
            visual_encoder,
            embed_visual=self.visual_embed,
            embed_tokens=self.shared,
        )
        self.encoder = VLBartEncoder(
            config=config,
            visual_encoder=self.visual_encoder,
            shared=self.shared,
        )
        self.decoder = VLBartDecoder(config, self.shared)
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, value):
        self.shared = value
        self.decoder.embed_tokens = self.shared

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def forward(
        self,
        input_pixels: torch.FloatTensor = None,
        node_ids=None,
        node_mask=None,
        obs_labels=None,
        matrix=None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, VLSeq2SeqModelOutput]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_pixels=input_pixels,
                node_ids=node_ids,
                node_mask=node_mask,
                matrix=matrix,
                obs_labels=obs_labels,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        cls_node_mask = encoder_outputs.attentions
        if cls_node_mask is not None:
            node_mask = node_mask.masked_fill(
                ((node_mask == 3)) & (cls_node_mask == 0), 0
            )
        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_visual_hidden_states=encoder_outputs.visual_last_hidden_state,
            encoder_node_hidden_states=encoder_outputs.tag_last_hidden_state,
            encoder_node_attention_mask=node_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        if not return_dict:
            return decoder_outputs + encoder_outputs

        return VLSeq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_visual_last_hidden_state=encoder_outputs.visual_last_hidden_state,
            encoder_visual_hidden_states=encoder_outputs.visual_hidden_states,
            encoder_tag_last_hidden_state=encoder_outputs.tag_last_hidden_state,
            encoder_attentions=encoder_outputs.attentions,
            node_cls_logits=encoder_outputs.node_cls_logits,
        )


class BartForConditionalGeneration(BartPretrainedModel):
    base_model_prefix = "model"
    _keys_to_ignore_on_load_missing = [r"final_logits_bias", r"lm_head\.weight"]

    def __init__(self, config: BartConfig, tokenizer: Tokenizer):
        super().__init__(config)
        self.visual_encoder = VisualExtractor(config)
        self.model = BartModel(config, self.visual_encoder)
        self.register_buffer(
            "final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings))
        )
        self.lm_head = nn.Linear(
            config.d_model,
            self.model.shared.num_embeddings,
            bias=False,
        )
        self.main_input_name = "pixel_values"
        self.tokenizer = tokenizer
        self.model.encoder.tokenizer = tokenizer
        # Initialize weights and apply final processing
        self.post_init()

    def get_encoder(self):
        return self.model.get_encoder()

    def get_decoder(self):
        return self.model.get_decoder()

    def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
        new_embeddings = super().resize_token_embeddings(new_num_tokens)
        self._resize_final_logits_bias(new_num_tokens)
        return new_embeddings

    def _resize_final_logits_bias(self, new_num_tokens: int) -> None:
        old_num_tokens = self.final_logits_bias.shape[-1]
        if new_num_tokens <= old_num_tokens:
            new_bias = self.final_logits_bias[:, :new_num_tokens]
        else:
            extra_bias = torch.zeros(
                (1, new_num_tokens - old_num_tokens),
                device=self.final_logits_bias.device,
            )
            new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
        self.register_buffer("final_logits_bias", new_bias)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def tie_weights(self):
        return super().tie_weights()

    def forward(
        self,
        input_pixels: torch.FloatTensor = None,
        node_ids=None,
        node_mask=None,
        ngram_labels=None,
        matrix=None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, VLSeq2SeqLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if labels is not None:
            if use_cache:
                logger.warning(
                    "The `use_cache` argument is changed to `False` since `labels` is provided."
                )
            use_cache = False
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels,
                    self.config.pad_token_id,
                    self.config.decoder_start_token_id,
                )
        outputs = self.model(
            input_pixels,
            node_ids=node_ids,
            node_mask=node_mask,
            obs_labels=ngram_labels,
            matrix=matrix,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(
                lm_logits.view(-1, self.config.vocab_size),
                labels.view(-1),
            )
            if ngram_labels is not None:
                ngram_mask = (ngram_labels != -1).float()
                ngram_labels = (
                    ngram_labels.masked_fill(ngram_mask == 0, 0).view(-1).float()
                )
                weight = torch.ones_like(ngram_labels) + ngram_labels * self.config.beta
                loss_fct = nn.BCEWithLogitsLoss(
                    reduction="none",
                    weight=weight,
                )
                node_cls_loss = loss_fct(
                    outputs.node_cls_logits.view(-1),
                    ngram_labels.view(-1).float(),
                ) * ngram_mask.view(-1)
                node_cls_loss = node_cls_loss.sum() / ngram_mask.sum()
                masked_lm_loss = masked_lm_loss + node_cls_loss
        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return (
                ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
            )

        return VLSeq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_visual_last_hidden_state=outputs.encoder_visual_last_hidden_state,
            encoder_visual_hidden_states=outputs.encoder_visual_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past=None,
        node_ids=None,
        node_mask=None,
        matrix=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs,
    ):
        # cut decoder_input_ids if past is used
        if past is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]
        return {
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "node_ids": node_ids,
            "node_mask": node_mask,
            "matrix": matrix,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return shift_tokens_right(
            labels, self.config.pad_token_id, self.config.decoder_start_token_id
        )

    @staticmethod
    def _reorder_cache(past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            # cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past += (
                tuple(
                    past_state.index_select(0, beam_idx)
                    for past_state in layer_past[:2]
                )
                + layer_past[2:],
            )
        return reordered_past

    @staticmethod
    def _expand_inputs_for_generation(
        input_ids: torch.LongTensor,  # decoder_input_ids
        expand_size: int = 1,
        is_encoder_decoder: bool = False,
        encoder_outputs: ModelOutput = None,
        **model_kwargs,
    ) -> Tuple[torch.LongTensor, Dict[str, Any]]:
        expanded_return_idx = (
            torch.arange(input_ids.shape[0])
            .view(-1, 1)
            .repeat(1, expand_size)
            .view(-1)
            .to(input_ids.device)
        )
        input_ids = input_ids.index_select(0, expanded_return_idx)

        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = token_type_ids.index_select(
                0, expanded_return_idx
            )

        if "attention_mask" in model_kwargs:
            attention_mask = model_kwargs["attention_mask"]
            model_kwargs["attention_mask"] = attention_mask.index_select(
                0, expanded_return_idx
            )

        if "node_ids" in model_kwargs:
            node_ids = model_kwargs["node_ids"]
            model_kwargs["node_ids"] = node_ids.index_select(0, expanded_return_idx)

        if "node_mask" in model_kwargs:
            node_mask = model_kwargs["node_mask"]
            model_kwargs["node_mask"] = node_mask.index_select(0, expanded_return_idx)

        if "matrix" in model_kwargs:
            matrix = model_kwargs["matrix"]
            model_kwargs["matrix"] = matrix.index_select(0, expanded_return_idx)

        if is_encoder_decoder:
            if encoder_outputs is None:
                raise ValueError(
                    "If `is_encoder_decoder` is True, make sure that `encoder_outputs` is defined."
                )
            if "tag_last_hidden_state" in encoder_outputs:
                encoder_outputs[
                    "tag_last_hidden_state"
                ] = encoder_outputs.tag_last_hidden_state.index_select(
                    0,
                    expanded_return_idx.to(
                        encoder_outputs.tag_last_hidden_state.device
                    ),
                )

            if "attentions" in encoder_outputs:
                encoder_outputs["attentions"] = encoder_outputs.attentions.index_select(
                    0, expanded_return_idx.to(encoder_outputs.attentions.device)
                )

            encoder_outputs[
                "visual_last_hidden_state"
            ] = encoder_outputs.visual_last_hidden_state.index_select(
                0,
                expanded_return_idx.to(encoder_outputs.visual_last_hidden_state.device),
            )
            model_kwargs["encoder_outputs"] = encoder_outputs
        return input_ids, model_kwargs
