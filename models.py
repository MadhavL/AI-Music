from absl import flags

import numpy as np
from absl import flags

import numpy as np
import torch
import torch.nn as nn
from math import sqrt

import torch.nn.functional
import base_models
from hparams import hparams_music_transformer, hparams_motion_transformer, hparams_cross_modal_transformer
from layers import DecoderLayer, abs_positional_encoding

HIDDEN_SIZE = 800
NHEAD_MOTION = 10
NHEAD_AUDIO = 10
NHEAD_CROSS = 10
NLAYERS_MOTION = 2
NLAYERS_AUDIO = 2
NLAYERS_CROSS = 12
MOTION_FEATURES = 96
AUDIO_FEATURES = 35
STD = 0.02

#######----------------------------------------------------------------------------------------------------------------#####

class PositionEmbedding(nn.Module):
    def __init__(self, seq_length, dim):
        super(PositionEmbedding, self).__init__()
        self.pos_embedding = nn.Parameter(torch.empty(seq_length, dim))
        nn.init.trunc_normal_(self.pos_embedding, std=STD)

    def forward(self, x):
        return x + self.pos_embedding

class fact_model_custom(nn.Module):
    def __init__(self, motion_input_len=120, audio_input_len=240, use_cuda=True):
        super().__init__()
        self.use_cuda = use_cuda
        self.motion_linear_embedding = nn.Linear(in_features=MOTION_FEATURES, out_features=HIDDEN_SIZE)
        self.audio_linear_embedding = nn.Linear(in_features=AUDIO_FEATURES, out_features=HIDDEN_SIZE)

        self.motion_position_embedding = PositionEmbedding(seq_length=motion_input_len, dim=HIDDEN_SIZE)
        self.audio_position_embedding = PositionEmbedding(seq_length=audio_input_len, dim=HIDDEN_SIZE)

        self.motion_transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=HIDDEN_SIZE, nhead=NHEAD_MOTION, dim_feedforward=HIDDEN_SIZE, activation="gelu")
        self.motion_transformer_encoder = nn.TransformerEncoder(encoder_layer=self.motion_transformer_encoder_layer, num_layers=NLAYERS_MOTION)
        self.audio_transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=HIDDEN_SIZE, nhead=NHEAD_AUDIO, dim_feedforward=HIDDEN_SIZE, activation="gelu")
        self.audio_transformer_encoder = nn.TransformerEncoder(encoder_layer=self.audio_transformer_encoder_layer, num_layers=NLAYERS_AUDIO)

        self.cross_modal_encoder_layer = nn.TransformerEncoderLayer(d_model=HIDDEN_SIZE, nhead=NHEAD_CROSS, dim_feedforward=HIDDEN_SIZE, activation="gelu")
        self.cross_modal_transformer_encoder = nn.TransformerEncoder(encoder_layer=self.cross_modal_encoder_layer, num_layers=NLAYERS_CROSS)

        self.cross_modal_output_layer = nn.Linear(in_features=HIDDEN_SIZE, out_features=MOTION_FEATURES)
        nn.init.trunc_normal_(self.cross_modal_output_layer.weight, std=STD)

        # MISSING:
        # Intermediate size in MLP / feedforward layer

    def forward(self, x):
        motion_input = x[0]
        audio_input = x[1]
        if self.use_cuda:
            motion_input = motion_input.cuda()
            audio_input = audio_input.cuda()
        motion_features = self.motion_linear_embedding(motion_input)
        motion_features = self.motion_position_embedding(motion_features)
        motion_features = self.motion_transformer_encoder(motion_features)

        audio_features = self.audio_linear_embedding(audio_input)
        audio_features = self.audio_position_embedding(audio_features)
        audio_features = self.audio_transformer_encoder(audio_features)

        try:
            assert(audio_features.shape[2] == motion_features.shape[2])
        except AssertionError:
            print(f"Output dimensions of audio & motion features don't match! Audio: {audio_features.shape}, motion: {motion_features.shape}")
        
        merged_sequences = torch.cat([motion_features, audio_features], dim=1)
        merged_sequences = self.cross_modal_transformer_encoder(merged_sequences)
        logits = self.cross_modal_output_layer(merged_sequences)

        return logits
    
#######----------------------------------------------------------------------------------------------------------------#####
######### ORIGINAL IMPLEMENTATION OF FACT MODEL #######

class FACTModel(nn.Module):
    """Audio Motion Multi-Modal model."""

    def __init__(self, use_cuda=True):
        """Initializer for FACTModel.

        Args:
            config: `FACTConfig` instance.
            is_training: bool. true for training model, false for eval model. Controls
                whether dropout will be applied.
        """
        super(FACTModel, self).__init__()

        # Build the cross modal transformer layer from the cross_modal_modal object in the config
        self.cross_modal_layer = base_models.CrossModalLayer()

        self.use_cuda = use_cuda

        # self.padding_factor = 0

        # Construct the motion and audio transformers and positional embeddings from the configs
        self.motion_transformer = base_models.Transformer(
            hidden_size=hparams_motion_transformer['d_model'],
            num_hidden_layers=hparams_motion_transformer['num_layers'],
            num_attention_heads=hparams_motion_transformer['num_heads'],
            intermediate_size=hparams_motion_transformer['intermediate_size'], #Confirm
            initializer_range=hparams_motion_transformer['initializer_range']) #Confirm
        self.motion_pos_embedding = base_models.PositionEmbedding(
            hparams_motion_transformer['motion_input_len'],
            hparams_motion_transformer['d_model'])
        self.motion_linear_embedding = base_models.LinearEmbedding(
            hparams_motion_transformer['motion_num_features'], 
            hparams_motion_transformer['d_model'])

        # Hardcoded for now
        self.audio_transformer = base_models.Transformer(
            hidden_size=800,
            num_hidden_layers=2,
            num_attention_heads=10,
            intermediate_size=800, #Confirm
            initializer_range=0.02)
        self.audio_pos_embedding = base_models.PositionEmbedding(
            240,
            800)
        self.audio_linear_embedding = base_models.LinearEmbedding(
            35, 800)

    def forward(self, inputs):
        """Predict sequences from inputs. 

        This is a single forward pass that been used during training. 

        Args:
            inputs: Input dict of tensors. The dict should contain 
                `motion_input` ([batch_size, motion_seq_length, motion_feature_dimension]) and
                `audio_input` ([batch_size, audio_seq_length, audio_feature_dimension]).

        Returns:
            Final output after the cross modal transformer. A tensor with shape 
            [batch_size, motion_seq_length + audio_seq_length, motion_feature_dimension]
            will be returned. **Be aware only the first N-frames are supervised during training**
        """
        # Computes motion features.
        motion_input = inputs[0]
        audio_input = inputs[1]

        # target = torch.zeros(audio_input.shape[0] + self.padding_factor, 360, 96)
        # target[:32, :, :] = audio_input

        if self.use_cuda:
            motion_input = motion_input.cuda()
            audio_input = audio_input.cuda()

        motion_features = self.motion_linear_embedding(motion_input)
        motion_features = self.motion_pos_embedding(motion_features)
        motion_features = self.motion_transformer(motion_features)

        # Computes audio features.
        audio_features = self.audio_linear_embedding(audio_input)
        audio_features = self.audio_pos_embedding(audio_features)
        audio_features = self.audio_transformer(audio_features)

        # Computes cross modal output.
        output = self.cross_modal_layer(motion_features, audio_features)
        # self.padding_factor += 1

        return output
    
#######----------------------------------------------------------------------------------------------------------------#####
## Original FACT Model modified for MIDI data

class FACTModel_midi(nn.Module):
    """Audio Motion Multi-Modal model."""

    def __init__(self, use_cuda=True):
        """Initializer for FACTModel.

        Args:
            config: `FACTConfig` instance.
            is_training: bool. true for training model, false for eval model. Controls
                whether dropout will be applied.
        """
        super(FACTModel_midi, self).__init__()

        # Build the cross modal transformer layer from the cross_modal_modal object in the config
        self.cross_modal_layer = base_models.CrossModalLayer()

        self.use_cuda = use_cuda

        # self.padding_factor = 0

        # Construct the motion and audio transformers and positional embeddings from the configs
        self.motion_transformer = base_models.Transformer(
            hidden_size=hparams_motion_transformer['d_model'],
            num_hidden_layers=hparams_motion_transformer['num_layers'],
            num_attention_heads=hparams_motion_transformer['num_heads'],
            intermediate_size=hparams_motion_transformer['intermediate_size'], #Confirm
            initializer_range=hparams_motion_transformer['initializer_range']) #Confirm
        self.motion_pos_embedding = base_models.PositionEmbedding(
            hparams_motion_transformer['motion_input_len'],
            hparams_motion_transformer['d_model'])
        self.motion_linear_embedding = base_models.LinearEmbedding(
            hparams_motion_transformer['motion_num_features'], 
            hparams_motion_transformer['d_model'])

        self.music_transformer = MusicTransformer()

    def forward(self, inputs):
        """Predict sequences from inputs. 

        This is a single forward pass that been used during training. 

        Args:
            inputs: Input dict of tensors. The dict should contain 
                `motion_input` ([batch_size, motion_seq_length, motion_feature_dimension]) and
                `midi_input` ([batch_size, midi_context_window).

        Returns:
            Final output after the cross modal transformer. A tensor with shape 
            [batch_size, motion_seq_length + audio_seq_length, motion_feature_dimension]
            will be returned. **Be aware only the first N-frames are supervised during training**
        """
        # Computes motion features.
        motion_input = inputs[0]
        midi_input = inputs[1]

        # target = torch.zeros(audio_input.shape[0] + self.padding_factor, 360, 96)
        # target[:32, :, :] = audio_input

        if self.use_cuda:
            motion_input = motion_input.cuda()
            midi_input = midi_input.cuda()

        motion_features = self.motion_linear_embedding(motion_input)
        motion_features = self.motion_pos_embedding(motion_features)
        motion_features = self.motion_transformer(motion_features)

        # Computes midi features.
        midi_features = self.music_transformer(midi_input)
        
        # Computes cross modal output.
        output = self.cross_modal_layer(motion_features, midi_features)
        # self.padding_factor += 1

        return output

#######----------------------------------------------------------------------------------------------------------------#####
### MUSIC TRANSFORMER MODEL
"""
Implementation of Music Transformer model, using torch.nn.TransformerDecoder
based on Huang et. al, 2018, Vaswani et. al, 2017
"""

### FROM https://github.com/spectraldoy/music-transformer/ 
### Can replace with different implementation

class MusicTransformer(nn.Module):
    """
    Transformer Decoder with Relative Attention. Consists of:
        1. Input Embedding
        2. Absolute Positional Encoding
        3. Stack of N DecoderLayers
        4. Final Linear Layer
    """
    def __init__(self,
                 d_model=hparams_music_transformer["d_model"],
                 num_layers=hparams_music_transformer["num_layers"],
                 num_heads=hparams_music_transformer["num_heads"],
                 d_ff=hparams_music_transformer["d_ff"],
                 max_rel_dist=hparams_music_transformer["max_rel_dist"],
                 max_abs_position=hparams_music_transformer["max_abs_position"],
                 vocab_size = hparams_music_transformer['vocab_size'],
                 bias=hparams_music_transformer["bias"],
                 dropout=hparams_music_transformer["dropout"],
                 layernorm_eps=hparams_music_transformer["layernorm_eps"]):
        """
        Args:
            d_model (int): Transformer hidden dimension size
            num_heads (int): number of heads along which to calculate attention
            d_ff (int): intermediate dimension of FFN blocks
            max_rel_dist (int): maximum relative distance between positions to consider in creating
                                relative position embeddings. Set to 0 to compute normal attention
            max_abs_position (int): maximum absolute position for which to create sinusoidal absolute
                                    positional encodings. Set to 0 to compute pure relative attention
                                    make it greater than the maximum sequence length in the dataset if nonzero
            bias (bool, optional): if set to False, all Linear layers in the MusicTransformer will not learn
                                   an additive bias. Default: True
            dropout (float in [0, 1], optional): dropout rate for training the model. Default: 0.1
            layernorm_eps (very small float, optional): epsilon for LayerNormalization. Default: 1e-6
        """
        super(MusicTransformer, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.max_rel_dist = max_rel_dist,
        self.max_position = max_abs_position
        self.vocab_size = vocab_size

        self.input_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = abs_positional_encoding(max_abs_position, d_model)
        self.input_dropout = nn.Dropout(dropout)

        self.decoder = nn.TransformerDecoder(
            DecoderLayer(d_model=d_model, num_heads=num_heads, d_ff=d_ff, max_rel_dist=max_rel_dist,
                         bias=bias, dropout=dropout, layernorm_eps=layernorm_eps),
            num_layers=num_layers,
            norm=nn.LayerNorm(normalized_shape=d_model, eps=layernorm_eps)
        )

        self.final = nn.Linear(d_model, hparams_cross_modal_transformer['d_model'])

    def forward(self, x, mask=None):
        """
        Forward pass through the Music Transformer. Embeds x according to Vaswani et. al, 2017, adds absolute
        positional encoding if present, performs dropout, passes through the stack of decoder layers, and
        projects into the vocabulary space. DOES NOT SOFTMAX OR SAMPLE OUTPUT; OUTPUTS LOGITS.

        Args:
            x (torch.Tensor): input batch of sequences of shape (batch_size, seq_len)
            mask (optional): mask for input batch indicating positions in x to mask with 1's. Default: None

        Returns:
            input batch after above steps of forward pass through MusicTransformer
        """
        # embed x according to Vaswani et. al, 2017
        x = self.input_embedding(x)
        x *= sqrt(self.d_model)

        # add absolute positional encoding if max_position > 0, and assuming max_position >> seq_len_x
        if self.max_position > 0:
            x += self.positional_encoding[:, :x.shape[-2], :]

        # input dropout
        x = self.input_dropout(x)

        # pass through decoder
        x = self.decoder(x, memory=None, tgt_mask=mask)

        # final projection to vocabulary space
        return self.final(x)

#######----------------------------------------------------------------------------------------------------------------#####
### Music Transformer Encoder only

class MusicTransformerEncoder(nn.Module):
    """
    Transformer Decoder with Relative Attention. Consists of:
        1. Input Embedding
        2. Absolute Positional Encoding
        3. Stack of N DecoderLayers
        4. Final Linear Layer
    """
    def __init__(self,
                 d_model=hparams_music_transformer["d_model"],
                 num_layers=hparams_music_transformer["num_layers"],
                 num_heads=hparams_music_transformer["num_heads"],
                 d_ff=hparams_music_transformer["d_ff"],
                 max_rel_dist=hparams_music_transformer["max_rel_dist"],
                 max_abs_position=hparams_music_transformer["max_abs_position"],
                 vocab_size=hparams_music_transformer["vocab_size"],
                 bias=hparams_music_transformer["bias"],
                 dropout=hparams_music_transformer["dropout"],
                 layernorm_eps=hparams_music_transformer["layernorm_eps"]):
        """
        Args:
            d_model (int): Transformer hidden dimension size
            num_heads (int): number of heads along which to calculate attention
            d_ff (int): intermediate dimension of FFN blocks
            max_rel_dist (int): maximum relative distance between positions to consider in creating
                                relative position embeddings. Set to 0 to compute normal attention
            max_abs_position (int): maximum absolute position for which to create sinusoidal absolute
                                    positional encodings. Set to 0 to compute pure relative attention
                                    make it greater than the maximum sequence length in the dataset if nonzero
            bias (bool, optional): if set to False, all Linear layers in the MusicTransformer will not learn
                                   an additive bias. Default: True
            dropout (float in [0, 1], optional): dropout rate for training the model. Default: 0.1
            layernorm_eps (very small float, optional): epsilon for LayerNormalization. Default: 1e-6
        """
        super(MusicTransformerEncoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.max_rel_dist = max_rel_dist,
        self.max_position = max_abs_position
        self.vocab_size = vocab_size

        self.input_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = abs_positional_encoding(max_abs_position, d_model)
        self.input_dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        Forward pass through the Music Transformer. Embeds x according to Vaswani et. al, 2017, adds absolute
        positional encoding if present, performs dropout, passes through the stack of decoder layers, and
        projects into the vocabulary space. DOES NOT SOFTMAX OR SAMPLE OUTPUT; OUTPUTS LOGITS.

        Args:
            x (torch.Tensor): input batch of sequences of shape (batch_size, seq_len)
            mask (optional): mask for input batch indicating positions in x to mask with 1's. Default: None

        Returns:
            input batch after above steps of forward pass through MusicTransformer
        """
        print(f"Input shape: {x.shape}")
        # embed x according to Vaswani et. al, 2017
        x = self.input_embedding(x)
        x *= sqrt(self.d_model)

        # add absolute positional encoding if max_position > 0, and assuming max_position >> seq_len_x
        if self.max_position > 0:
            x += self.positional_encoding[:, :x.shape[-2], :]

        # input dropout
        x = self.input_dropout(x)

        return x