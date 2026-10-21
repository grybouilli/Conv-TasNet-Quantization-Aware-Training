from typing import Generator

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from model.norm import GlobalLayerNorm, NormType, fold_conv_bn
from convtasnet import ConvTasNet, ConvolutionalBlock1D, TCN

#TODO: Bufferize the streamable versions
class StreamableConvolutionalBlock1D(nn.Module):
    """
    [Batch, B, L] -> [Batch, H, L] -> output: [Batch, B, L]
                                      skip:   [Batch, Sc, L]

    Args:
        in_channels: `B`, number of input channels
        hidden_channels: number of channels in the hidden layers of this convolutional block
        skip_channels: `Sc`, number of channels in the skip connection output
        kernel_size: kernel size of the depth-wise convolution
        padding: padding of the depth-wise convolution
        dilation: dilation of the depth-wise convolution

    B correspond to the number of channels in bottleneck and the residual paths
    """

    def __init__(
        self,
        base : ConvolutionalBlock1D
    ):
        super().__init__()

        self.conv1d = fold_conv_bn(base.conv1d, base.normalization1)
        self.nonlinearity1 = nn.PReLU()

        self.dconv = fold_conv_bn(base.dconv, base.normalization2)
        self.nonlinearity2 = nn.PReLU()

        self.base = base

    def forward(self, input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            input: a [Batch, B, L] tensor

        Returns:
            A tuple containing the output tensor and the skip connection tensor
            output: [Batch, B, L]
            skip:   [Batch, Sc, L]
        """
        output = self.conv1d(input)
        output = self.nonlinearity1(output)

        output = self.dconv(output)
        output = self.nonlinearity2(output)

        skip = self.base.skip_conv1d(output)
        residual = self.base.out_conv1d(output)

        # TODO: we put this here instead of in the TCN which may be unconventional
        output = input + residual
        return output, skip


class StreamableTCN(nn.Module):
    """
    Temporal Convolutional Network (TCN)
    Fully-convolutional separation module that consists of stacked 1-D dilated convolutional blocks
    Dilation factors increase exponentially: M convolutional blocks with dilation factors 1,2,4,...,2^(M-1) are repeated R times

    Return masks of shape [Batch, n_src, N, L]

    bottleneck_dim: `B`, dimension of the bottleneck layer

    [Batch, N, L] -> [Batch, n_src, N, L]

    Args:
        input_dim: `N`, dimension of the input layer, correspond to the number of filters in autoencoder
        n_src: number of speakers/sources to separate
        bottleneck_dim: `B`, dimension of the bottleneck layer
        hidden_dim: `H`, dimension of the hidden layers
        skip_dim: `Sc`, dimension of the skip connection layer
        kernel_size: `P`, kernel size of the depth-wise convolution
        nb_layers: `R`, number of layers
        nb_conv_blocks_per_layer: `X` (`M` in equations), number of convolutional blocks per layer
        norm_type: type of normalization to use in the convolutional blocks
            global layer norm or channel-wise layer norm
            gln usually works better for non-causal configuration.

    """

    def __init__(
        self,
        base : TCN
    ):
        super().__init__()
        self.base = base
        self.layers: nn.ModuleList[StreamableConvolutionalBlock1D] = nn.ModuleList(modules=[])

        for layer in base.layers:
            self.layers.append(StreamableConvolutionalBlock1D(layer))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        batch_size, _, length_L = input.size()
        # input shape: [Batch, N, L]
        normed_input = self.base.layer_norm(input)
        # assert normed_input.shape == input.shape
        # normed_input shape: [Batch, N, L]
        bn_input = self.base.bottleneck_layer(normed_input)
        # bn_input shape: [Batch, B, L]
        conv_block_input = bn_input

        skip_connection_sum = torch.zeros(
            (conv_block_input.size(0), self.base.skip_dim, conv_block_input.size(2)),
            device=conv_block_input.device,
        )
        # skip_connection_sum shape: [Batch, Sc, L]

        for conv_block_layer in self.layers:
            conv_block_output, skip = conv_block_layer(conv_block_input)
            # conv_block_output shape: [Batch, B, L]
            # skip shape: [Batch, Sc, L]
            conv_block_input = conv_block_output
            skip_connection_sum = skip_connection_sum + skip

        output = self.base.non_linearity_out(skip_connection_sum)
        # output: [Batch, Sc, L]
        output = self.base.conv1d_out(output)
        # output: [Batch, N * n_src, L]
        output = self.base.mask_activation(output)
        # output: [Batch, N * n_src, L]
        masks = output.view(batch_size, self.base.n_src, self.base.input_dim, length_L)
        # masks: [Batch, n_src, N, L]
        return masks


class StreamableConvTasNet(nn.Module):
    def __init__(self, base : ConvTasNet):
        super().__init__()
        
        self.base = base
        self.tcn = StreamableTCN(base.tcn)
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
            # Input: [Batch, L] or [Batch, 1, L]
    
            # First we pad
            padded_input, rest = self.base.pad_signal(input)
    
            enc_output_mixture = self.base.encoder(padded_input)
            # enc_output_mixture shape: [Batch, N, L']
    
            tcn_output_masks = self.tcn(enc_output_mixture)
            # tcn_output_masks shape: [Batch, n_src, N, L']
    
            # We apply the masks
            # enc_output_mixture.unsqueeze(1) shape: [Batch, 1, N, L']
            masked_output = tcn_output_masks * enc_output_mixture.unsqueeze(1)
            # masked_output shape: [Batch, n_src, N, L']
    
            # We need to reshape the masked output to apply the decoder on each source separately
            batch_size, n_src, enc_dim, length_L = masked_output.size()
            masked_output = masked_output.view(batch_size * n_src, enc_dim, length_L)
            # masked_output shape: [Batch * n_src, N, L']
            # we treat each source as a separate sample in the batch for the decoder
    
            dec_output = self.base.decoder(masked_output)
            # dec_output shape: [Batch * n_src, 1, L]
    
            # We remove extra
            # See the paper implementation
            output = dec_output[:, :, self.base.stride : -(rest + self.base.stride)].contiguous()
            output = output.view(batch_size, n_src, -1)
            # output shape: [Batch, n_src, L]
            return output
