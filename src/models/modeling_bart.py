# coding=utf-8
# Copyright 2021 The Fairseq Authors and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch BART model."""
import copy
import math
import random
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
import torchvision.models as models
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.models.bart.configuration_bart import BartConfig
from transformers.models.bart.modeling_bart import BartAttention, BartPretrainedModel
from transformers.utils import logging

logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "facebook/bart-base"
_CONFIG_FOR_DOC = "BartConfig"
_TOKENIZER_FOR_DOC = "BartTokenizer"

# Base model docstring
_EXPECTED_OUTPUT_SHAPE = [1, 8, 768]

# SequenceClassification docstring
_CHECKPOINT_FOR_SEQUENCE_CLASSIFICATION = "valhalla/bart-large-sst2"
_SEQ_CLASS_EXPECTED_LOSS = 0.0
_SEQ_CLASS_EXPECTED_OUTPUT = "'POSITIVE'"

# QuestionAsnwering docstring
_CHECKPOINT_FOR_QA = "valhalla/bart-large-finetuned-squadv1"
_QA_EXPECTED_LOSS = 0.59
_QA_EXPECTED_OUTPUT = "' nice puppet'"

BART_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/bart-large",
    # see all BART models at https://huggingface.co/models?filter=bart
]


def shift_tokens_right(
    input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int
):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids


def _make_causal_mask(
    input_ids_shape: torch.Size, dtype: torch.dtype, past_key_values_length: int = 0
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), float("-inf"))
    mask_cond = torch.arange(mask.size(-1))
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat(
            [
                torch.zeros(
                    tgt_len,
                    past_key_values_length,
                    dtype=dtype,
                ),
                mask,
            ],
            dim=-1,
        )
    return mask[None, None, :, :].expand(
        bsz,
        1,
        tgt_len,
        tgt_len + past_key_values_length,
    )


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.bool(), torch.finfo(dtype).min)


class BartLearnedPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int):
        # Bart is set up so that if padding_idx is specified then offset the embedding ids by 2
        # and adjust num_embeddings appropriately. Other models don't have this hack
        self.offset = 2
        super().__init__(num_embeddings + self.offset, embedding_dim)

    def forward(self, input_ids_shape: torch.Size, past_key_values_length: int = 0):
        """`input_ids_shape` is expected to be [bsz x seqlen]."""
        bsz, seq_len = input_ids_shape[:2]
        positions = torch.arange(
            past_key_values_length,
            past_key_values_length + seq_len,
            dtype=torch.long,
            device=self.weight.device,
        )
        return super().forward(positions + self.offset)


class BartDecoderLayer(nn.Module):
    def __init__(self, config: BartConfig):
        super().__init__()
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
        self.encoder_attn = BartAttention(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
        )
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        cross_attn_layer_head_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = True,
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
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

        # Cross-Attention Block
        cross_attn_present_key_value = None
        cross_attn_weights = None
        if encoder_hidden_states is not None:
            residual = hidden_states

            # cross_attn cached key/values tuple is at positions 3,4 of present_key_value tuple
            cross_attn_past_key_value = (
                past_key_value[-2:] if past_key_value is not None else None
            )
            (
                hidden_states,
                cross_attn_weights,
                cross_attn_present_key_value,
            ) = self.encoder_attn(
                hidden_states=hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=cross_attn_past_key_value,
                output_attentions=output_attentions,
            )
            hidden_states = nn.functional.dropout(
                hidden_states, p=self.dropout, training=self.training
            )
            hidden_states = residual + hidden_states
            hidden_states = self.encoder_attn_layer_norm(hidden_states)

            # add cross-attn to positions 3,4 of present_key_value tuple
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


class GuidedBartEncoderLayer(nn.Module):
    def __init__(self, config: BartConfig):
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = BartAttention(
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            dropout=config.attention_dropout,
        )
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.self_obs_attn = BartAttention(
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            dropout=config.attention_dropout,
        )
        self.self_obs_attn_layer_norm = nn.LayerNorm(self.embed_dim)

        self.v2o_attn = BartAttention(
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            dropout=config.attention_dropout,
        )
        self.v2o_attn_layer_norm = nn.LayerNorm(self.embed_dim)

        self.o2v_attn = BartAttention(
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            dropout=config.attention_dropout,
        )
        self.o2v_attn_layer_norm = nn.LayerNorm(self.embed_dim)

        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

        self.obs_fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.obs_fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_obs_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        obs_hidden_states: torch.Tensor,
        obs_attention_mask: torch.Tensor,
        cross_attention_mask: torch.Tensor,
        layer_head_mask: torch.Tensor,
        output_attentions: bool = False,
    ):
        residual = hidden_states
        hidden_states, attn_weights, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.dropout, training=self.training
        )
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        obs_residual = obs_hidden_states
        obs_hidden_states, attn_weights, _ = self.self_obs_attn(
            hidden_states=obs_hidden_states,
            attention_mask=obs_attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        obs_hidden_states = nn.functional.dropout(
            obs_hidden_states, p=self.dropout, training=self.training
        )
        obs_hidden_states = obs_residual + obs_hidden_states
        obs_hidden_states = self.self_obs_attn_layer_norm(obs_hidden_states)

        residual = hidden_states
        guided_hidden_states, _, _ = self.v2o_attn(
            hidden_states=hidden_states,
            key_value_states=obs_hidden_states,
            attention_mask=cross_attention_mask,
        )
        guided_hidden_states = nn.functional.dropout(
            guided_hidden_states, p=self.dropout, training=self.training
        )
        guided_hidden_states = residual + guided_hidden_states
        guided_hidden_states = self.v2o_attn_layer_norm(guided_hidden_states)

        residual = obs_hidden_states
        guided_obs_hidden_states, _, _ = self.o2v_attn(
            hidden_states=obs_hidden_states,
            key_value_states=hidden_states,
            attention_mask=cross_attention_mask.transpose(3, 2),
        )
        guided_obs_hidden_states = nn.functional.dropout(
            guided_obs_hidden_states, p=self.dropout, training=self.training
        )
        guided_obs_hidden_states = residual + guided_obs_hidden_states
        guided_obs_hidden_states = self.o2v_attn_layer_norm(guided_obs_hidden_states)

        residual = guided_hidden_states
        hidden_states = self.activation_fn(self.fc1(guided_hidden_states))
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.activation_dropout, training=self.training
        )
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.dropout, training=self.training
        )
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        residual = guided_obs_hidden_states
        obs_hidden_states = self.activation_fn(self.obs_fc1(guided_obs_hidden_states))
        obs_hidden_states = nn.functional.dropout(
            obs_hidden_states, p=self.activation_dropout, training=self.training
        )
        obs_hidden_states = self.obs_fc2(obs_hidden_states)
        obs_hidden_states = nn.functional.dropout(
            obs_hidden_states, p=self.dropout, training=self.training
        )
        obs_hidden_states = residual + obs_hidden_states
        obs_hidden_states = self.final_obs_layer_norm(obs_hidden_states)

        if hidden_states.dtype == torch.float16 and (
            torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()
        ):
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(
                hidden_states, min=-clamp_value, max=clamp_value
            )
            obs_hidden_states = torch.clamp(
                obs_hidden_states, min=-clamp_value, max=clamp_value
            )

        outputs = (
            hidden_states,
            obs_hidden_states,
        )

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class VisualBartEncoder(BartPretrainedModel):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    [`BartEncoderLayer`].

    Args:
        config: BartConfig
        embed_tokens (nn.Embedding): output embedding
    """

    def __init__(
        self,
        config: BartConfig,
        visual_encoder,
        embed_visual=None,
        embed_tokens: Optional[nn.Embedding] = None,
    ):
        super().__init__(config)

        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop

        embed_dim = config.d_model
        self.padding_idx = config.pad_token_id
        self.visual_encoder = visual_encoder
        self.embed_visual = embed_visual
        self.embed_tokens = embed_tokens

        from transformers.models.bart.modeling_bart import BartEncoderLayer

        self.layers = nn.ModuleList(
            [BartEncoderLayer(config) for _ in range(config.encoder_layers)]
        )
        self.layernorm_embedding = nn.LayerNorm(embed_dim)
        # self.cls_node_head = nn.Linear(config.d_visual, config.mention_size)
        # self.obs_layernorm_embedding = nn.LayerNorm(embed_dim)
        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_pixels=None,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        outline_ids=None,
        outline_mask=None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
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
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        input_pixels_size = input_pixels.size()
        flatten_input_pixels = input_pixels.view(
            -1,
            *input_pixels_size[-3:],
        )
        visual_features = self.visual_encoder(flatten_input_pixels)
        visual_features = visual_features.reshape(
            input_pixels_size[0],
            -1,
            visual_features.size(-1),
        )
        hidden_states = self.embed_visual(visual_features)

        src_seq_len = hidden_states.size(1)
        attention_mask = None
        if outline_ids is not None:
            attention_mask = outline_mask.new_ones(*hidden_states.size()[:-1])
            attention_mask = torch.cat((attention_mask, outline_mask), dim=1)
            hidden_states = torch.cat((hidden_states, outline_ids), dim=1)

        unexpand_attention_mask = attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            dtype = hidden_states.dtype
            attention_mask = _expand_mask(attention_mask, dtype)
            attention_mask[..., :src_seq_len, src_seq_len:] = torch.finfo(dtype).min
            eye_mask = torch.eye(outline_ids.size(1), device=attention_mask.device)
            inverted_mask = 1.0 - eye_mask
            inverted_mask = inverted_mask.masked_fill(
                inverted_mask.bool(),
                torch.finfo(dtype).min,
            )
            attention_mask[..., src_seq_len:, src_seq_len:].masked_fill_(
                (attention_mask[..., src_seq_len:, src_seq_len:] != 0)
                | (inverted_mask != 0),
                torch.finfo(dtype).min,
            )

        hidden_states = self.layernorm_embedding(hidden_states)
        hidden_states = nn.functional.dropout(
            hidden_states,
            p=self.dropout,
            training=self.training,
        )

        encoder_states = () if output_hidden_states else None
        encoder_residual = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        # check if head_mask has a correct number of layers specified if desired
        if head_mask is not None:
            if head_mask.size()[0] != (len(self.layers)):
                raise ValueError(
                    f"The head_mask should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."
                )
        residual = None
        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
                if idx > 0:
                    encoder_residual = encoder_residual + (residual,)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)
            if self.training and (
                dropout_probability < self.layerdrop
            ):  # skip the layer
                layer_outputs = (None, None)
            else:
                if self.gradient_checkpointing and self.training:

                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs, output_attentions)

                        return custom_forward

                    layer_outputs = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(encoder_layer),
                        hidden_states,
                        attention_mask,
                        (head_mask[idx] if head_mask is not None else None),
                    )
                else:
                    layer_outputs = encoder_layer(
                        hidden_states,
                        attention_mask,
                        layer_head_mask=(
                            head_mask[idx] if head_mask is not None else None
                        ),
                        output_attentions=output_attentions,
                    )

                hidden_states = layer_outputs[0]
                # residual = layer_outputs[1]
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[-1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, encoder_states, all_attentions]
                if v is not None
            )

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            attentions=unexpand_attention_mask,
        )


class OutlineGraphEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        outline_offset: int,
    ):
        # plus 1 for padding_idx
        self.offset = 1
        self.outline_offset = outline_offset
        self.max_position = num_embeddings + outline_offset + self.offset
        super().__init__(
            self.max_position,
            embedding_dim,
        )

    def forward(
        self,
        input_ids_shape: torch.Size,
        past_key_values_length: int = 0,
        position_mask: Optional[torch.LongTensor] = None,
    ):
        """`input_ids_shape` is expected to be [bsz x seqlen]."""
        bsz, seq_len = input_ids_shape[:2]
        positions = (
            torch.arange(
                past_key_values_length,
                past_key_values_length + seq_len,
                dtype=torch.long,
                device=self.weight.device,
            )
            + self.offset
        )
        if position_mask is not None:
            positions = positions * position_mask
        return super().forward(positions)

    def get_level_embed(self, level):
        level = level + self.offset
        return self.weight[level : level + 1]


class GraphBartEncoderLayer(nn.Module):
    def __init__(self, config: BartConfig):
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = BartAttention(
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            dropout=config.attention_dropout,
        )
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: torch.FloatTensor,
        encoder_hidden_state: torch.FloatTensor,
        layer_head_mask: torch.FloatTensor,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor, Optional[torch.FloatTensor]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(seq_len, batch, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (`torch.FloatTensor`): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states
        hidden_states, attn_weights, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.dropout, training=self.training
        )
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

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

        if hidden_states.dtype == torch.float16 and (
            torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()
        ):
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(
                hidden_states, min=-clamp_value, max=clamp_value
            )

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class GraphBartEncoder(BartPretrainedModel):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    [`BartEncoderLayer`].

    Args:
        config: BartConfig
        embed_tokens (nn.Embedding): output embedding
    """

    def __init__(
        self,
        config: BartConfig,
        embed_tokens: nn.Embedding,
    ):
        super().__init__(config)
        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop
        embed_dim = config.d_model
        self.padding_idx = config.pad_token_id
        self.config = config
        self.dropout = config.dropout
        self.node_size = config.node_size
        self.outline_offset = max(0, config.outline_level - 1)  # for outline level 0
        max_outline_positions = (
            config.tag_size + self.outline_offset
        )  # 2 for ngram nodes and token nodes
        if max_outline_positions % 8 != 0:
            max_outline_positions = (max_outline_positions // 8 + 1) * 8
        self.embed_positions = OutlineGraphEmbedding(
            max_outline_positions,
            config.d_model,
            self.outline_offset,
        )
        self.node_embed = nn.Embedding(
            config.node_size + 1,
            config.d_model,
            config.node_size,
        )
        from transformers.models.bart.modeling_bart import BartEncoderLayer

        self.layers = nn.ModuleList(
            [BartEncoderLayer(config) for _ in range(config.rgcn_layers)]
        )
        self.layernorm_embedding = nn.LayerNorm(embed_dim)
        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        node_ids=None,
        node_embeds=None,
        node_mask=None,
        attention_mask=None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if node_embeds is None:
            node_embeds = self.node_embed(node_ids)
        position_mask = (node_mask == 1).float()
        positions = self.embed_positions(
            node_ids.size(),
            self.outline_offset,
            position_mask.long(),
        )
        positions = positions * position_mask.unsqueeze(-1)
        for level in range(self.outline_offset):  # outline_offset = 2
            level_mask = (
                node_mask == level + self.outline_offset
            ).float()  # level + 2, e.g., level 2 -> 2
            level_embed = (
                self.embed_positions.get_level_embed(level)
                .expand(level_mask.size(1), -1)
                .unsqueeze(0)
            )
            masked_level_embed = level_mask.unsqueeze(-1) * level_embed
            positions = positions + masked_level_embed

        hidden_states = node_embeds + positions
        inverted_mask = attention_mask == 0
        attention_mask = (
            inverted_mask.float()
            .masked_fill(inverted_mask, torch.finfo(hidden_states.dtype).min)
            .unsqueeze(1)
        )
        hidden_states = self.layernorm_embedding(hidden_states)
        hidden_states = nn.functional.dropout(
            hidden_states,
            p=self.dropout,
            training=self.training,
        )

        for idx, encoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)
            if self.training and (
                dropout_probability < self.layerdrop
            ):  # skip the layer
                layer_outputs = (None, None)
            else:
                layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask,
                    layer_head_mask=None,
                )
                hidden_states = layer_outputs[0]
        # hidden_states = hidden_states[:, encoder_hidden_state.size(1):]
        if not return_dict:
            return tuple(v for v in [hidden_states] if v is not None)

        return BaseModelOutput(
            last_hidden_state=hidden_states,
        )


class VisualEmbedding(nn.Module):
    def __init__(self, config):
        super(VisualEmbedding, self).__init__()
        self.embed = nn.Sequential(
            nn.Linear(config.d_visual, config.d_model),
            nn.ReLU(),
            nn.Dropout(config.dropout),
        )

    def forward(self, inputs):
        return self.embed(inputs)


class VisualExtractor(nn.Module):
    def __init__(self, config):
        super(VisualExtractor, self).__init__()
        self.visual_extractor = config.visual_extractor
        self.pretrained = config.visual_extractor_pretrained
        try:
            # if config.chexpert_model_name_or_path is not None:
            print("====================================================")
            print("====================================================")
            print(
                "Loading CheXpert Pretrained model: %s"
                % config.chexpert_model_name_or_path
            )
            print("====================================================")
            print("====================================================")
            chexpert_state_dict = torch.load(
                config.chexpert_model_name_or_path,
            )["state_dict"]
            model = getattr(
                models,
                self.visual_extractor,
            )(pretrained=False, num_classes=config.obs_num // 2)
            model.load_state_dict(chexpert_state_dict)
        except Exception as e:
            model = getattr(
                models,
                self.visual_extractor,
            )(pretrained=self.pretrained)
        modules = list(model.children())
        self.model = nn.Sequential(*modules[:-2])

    def forward(self, images):
        patch_feats = self.model(images)
        batch_size, feat_size, _, _ = patch_feats.shape
        patch_feats = patch_feats.reshape(
            batch_size,
            feat_size,
            -1,
        ).permute(0, 2, 1)
        return patch_feats
