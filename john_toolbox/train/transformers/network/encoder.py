import logging

import torch
from torch import nn

from john_toolbox.train.transformers.network.layers.attention import MultiHeadAttentionBlock
from john_toolbox.train.transformers.network.layers.feed_forward import FeedForwardBlock
from john_toolbox.train.transformers.network.layers.normalization import LayerNormalization
from john_toolbox.train.transformers.network.layers.residual import ResidualConnection

LOGGER = logging.getLogger(__name__)


class EncoderBlock(nn.Module):
    """docstring for EncoderBlock."""

    def __init__(
        self,
        features: int,
        self_attention_block: MultiHeadAttentionBlock,
        feed_forward_block: FeedForwardBlock,
        dropout: float,
    ):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList(
            [ResidualConnection(features=features, dropout=dropout) for _ in range(2)]
        )

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor):
        """
        Defines the forward pass of the EncoderBlock.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor to the encoder block. Typically, this is the output from the
            previous encoder block or the initial embedded input.
        src_mask : torch.Tensor
            A mask tensor used in the self-attention mechanism to prevent attention to
            certain positions, typically padding tokens.

        Returns
        -------
        torch.Tensor
            The output of the encoder block after applying self-attention and feed-forward
            network with residual connections.
        """
        # the src mask is the mask we want to apply to the input of the encoder.
        # we need this because we want to hide the interaction of the padding word
        # we don't want the padding word to interact with other words
        # ! Apply self-attention with residual connection
        x = self.residual_connections[0](
            x, lambda x: self.self_attention_block(q=x, k=x, v=x, mask=src_mask)
        )
        # In this line, a residual connection is applied around the self-attention block.
        # Residual connections help in avoiding the vanishing gradient problem by allowing
        # an alternate shortcut path for the gradient. The 'self_attention_block' is where
        # the self-attention mechanism is implemented. 'q', 'k', and 'v' represent query, key,
        # and value, respectively. In self-attention, these are all the same (i.e., 'x'),
        # meaning the model is processing its own input to determine the importance of each part.

        # The term 'self-attention' is used because the mechanism is applied to the same sequence,
        # meaning the sequence is essentially 'looking at' or analyzing itself to determine
        # the relative importance of its different parts.

        # In the decoder block of a Transformer model, a different type of attention called
        # 'cross-attention' is used. Here, the queries come from the decoder but they attend to
        # the keys and values from the encoder. This allows the decoder to 'look at' or integrate
        # information from the encoder. This is crucial in tasks like machine translation,
        # where the decoder needs to consider the context provided by the encoder to generate
        # an appropriate output.
        # ! Apply feed-forward network with residual connection
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x


class Encoder(nn.Module):
    """Encoder class."""

    def __init__(self, features: int, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features=features)

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        for layer in self.layers:
            x = layer(x, mask)

        return self.norm(x)
