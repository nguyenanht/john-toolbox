import logging

import torch
from torch import nn

from john_toolbox.train.transformers.network.layers.attention import MultiHeadAttentionBlock
from john_toolbox.train.transformers.network.layers.feed_forward import FeedForwardBlock
from john_toolbox.train.transformers.network.layers.normalization import LayerNormalization
from john_toolbox.train.transformers.network.layers.residual import ResidualConnection

LOGGER = logging.getLogger(__name__)


class DecoderBlock(nn.Module):
    """
    A Decoder Block in a Transformer model.

    This block is a part of the decoder side of the Transformer architecture.
    It consists of a self-attention block, a cross-attention block, and a feed-forward block, each followed by a residual connection.

    Parameters
    ----------
    self_attention_block : MultiHeadAttentionBlock
        The self-attention block for the decoder. It computes attention over the decoder's input.
    cross_attention_block : MultiHeadAttentionBlock
        The cross-attention block for the decoder. It computes attention over the output of the encoder.
    feed_forward_block : FeedForwardBlock
        The feed-forward block for the decoder. It applies position-wise feed-forward operations.
    dropout : float
        The dropout rate used in the residual connections.

    Methods
    -------
    forward(x, encoder_output, src_mask, tgt_mask)
        Defines the computation performed at every call. Requires the target sequence, encoder output, and corresponding masks.
    """

    def __init__(
        self,
        features: int,
        self_attention_block: "MultiHeadAttentionBlock",
        cross_attention_block: "MultiHeadAttentionBlock",
        feed_forward_block: "FeedForwardBlock",
        dropout: float,
    ):
        super().__init__()

        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        # Creating residual connections for each sub-block
        self.residual_connections = nn.ModuleList(
            [ResidualConnection(features=features, dropout=dropout) for _ in range(3)]
        )

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass for the Decoder Block.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor to the decoder block.
        encoder_output : torch.Tensor
            The output tensor from the encoder block.
        src_mask : torch.Tensor
            The mask for the encoder's output, to mask out padding tokens.
        tgt_mask : torch.Tensor
            The mask for the decoder's input, to prevent the model from attending to future tokens.

        Returns
        -------
        torch.Tensor
            The output tensor after passing through the decoder block.
        """
        # ! Apply self-attention and add the result to the input (residual connection)
        # Using lambda to delay the execution of the function in the residual connection.
        # This allows passing the function as a parameter and executing it with the
        # latest variable values only when necessary.
        x = self.residual_connections[0](
            x, lambda x: self.self_attention_block(q=x, k=x, v=x, mask=tgt_mask)
        )
        # ! Apply cross-attention with encoder output and add the result to the previous output (residual connection)
        # Similarly, lambda is used here to delay the execution of the cross_attention_block function.
        x = self.residual_connections[1](
            x,
            lambda x: self.cross_attention_block(
                q=x, k=encoder_output, v=encoder_output, mask=src_mask
            ),
        )
        # ! Apply the feed-forward network and add the result to the previous output (residual connection)
        x = self.residual_connections[2](
            x,
            self.feed_forward_block,
        )
        return x


class Decoder(nn.Module):
    """docstring for Decoder."""

    def __init__(self, features: int, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features=features)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(
                x=x,
                encoder_output=encoder_output,
                src_mask=src_mask,
                tgt_mask=tgt_mask,
            )
        return self.norm(x)
