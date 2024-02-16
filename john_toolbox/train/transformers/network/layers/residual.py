import logging

from torch import nn

from john_toolbox.train.transformers.network.layers.normalization import LayerNormalization

LOGGER = logging.getLogger(__name__)


class ResidualConnection(nn.Module):
    """
    A ResidualConnection module for a Transformer model.

    This module implements the residual connection used in Transformer models.
    Conceptually, the role of a residual connection is to facilitate training of
    deep neural networks by allowing gradients to flow through a network more
    effectively. In the context of Transformers, it helps in stabilizing the
    learning process and mitigates the vanishing gradient problem.

    The ResidualConnection achieves this by applying layer normalization to the input,
    followed by a specified sublayer (self-attention or feed-forward network), and then
    adds the output of this sublayer to the original input (creating the residual
    connection). Dropout is applied after the sublayer for regularization.

    This architecture ensures that the deeper layers of the network can continue to
    learn from the original input, as the unaltered input is carried forward directly.
    The combination of the original input and the output from the sublayer provides a
    "shortcut" for the gradient during backpropagation, which can help in training deeper
    networks more effectively.

    Parameters
    ----------
    dropout : float
        The dropout rate to be used after the sublayer.

    Attributes
    ----------
    dropout : nn.Dropout
        Dropout layer with the specified dropout rate.
    norm : LayerNormalization
        Layer normalization module.

    Methods
    -------
    forward(x, sublayer):
        Compute the output of the ResidualConnection module.

        Parameters:
        x : Tensor
            The input tensor to the ResidualConnection module.
        sublayer : nn.Module
            The sublayer to be applied, which could be a self-attention or a
            feed-forward network.

        Returns:
        Tensor
            The output tensor after applying the residual connection.
    """

    def __init__(self, features: int, dropout: float):
        """Initializes the ResidualConnection module.

        Args:
            dropout (float): The dropout rate to be used after the sublayer.
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.norm = LayerNormalization(features=features)  # Layer normalization module

    def forward(self, x, sublayer):
        """
        The forward pass of the ResidualConnection module.

        Parameters
        ----------
        x : Tensor
            The input tensor to the ResidualConnection module.
        sublayer : nn.Module
            The sublayer to be applied, which could be a self-attention or a
            feed-forward network.

        Returns
        -------
        Tensor
            The output tensor after applying the residual connection.
        """
        # The layer normalization is applied first to the input (x).
        # Then, the sublayer is applied to this normalized input.
        # Dropout is applied to the output of the sublayer for regularization.
        # Finally, the original input (x) is added to the output of the dropout
        # layer, creating the residual connection.
        # This process helps in stabilizing the learning and avoids the vanishing
        # gradient problem in deep networks.
        if x is None:
            LOGGER.error("x is None")
            raise ValueError("x is None")
        return x + self.dropout(sublayer(self.norm(x)))
        # Note: In the original Transformer paper, the norm is applied before the sublayer.
        # However, most implementations apply the sublayer first, then the norm.
