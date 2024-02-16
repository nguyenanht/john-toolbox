import torch.nn.functional as F
from torch import nn


class FeedForwardBlock(nn.Module):
    """
    A feed-forward block in a transformer model, consisting of two linear transformations
    with a ReLU activation in between.

    The primary role of this block is to provide additional representational power to the
    Transformer model. Each position in the sequence is passed through this fully connected
    neural network, enabling the model to integrate and learn from the entire sequence.
    While the attention mechanism captures positional dependencies, this feed-forward block
    applies complex, non-linear transformations to these representations, adding a crucial
    layer of abstraction and complexity. This enhances the Transformer's ability to learn
    more sophisticated representations and improves its overall predictive performance.

    Parameters
    ----------
    emb_size : int
        The size of the input and output embeddings. This is the dimensionality of the
        input and output of this block.

    d_ff : int
        The dimensionality of the hidden layer in the feed-forward network. This is
        typically larger than `emb_size`.

    dropout : float
        Dropout rate to use in the block. Dropout is a regularization technique to prevent
        overfitting.

    Attributes
    ----------
    linear1 : nn.Linear
        The first linear transformation layer which maps the input from `emb_size` to `d_ff`.

    dropout : nn.Dropout
        Dropout layer applied after the first linear transformation and activation.

    linear2 : nn.Linear
        The second linear transformation layer which maps the output of the dropout layer
        back from `d_ff` to `emb_size`.

    Methods
    -------
    forward(x)
        Defines the forward pass of the FeedForwardBlock.
    """

    def __init__(self, emb_size: int, d_ff: int, dropout: float):
        super().__init__()
        # First linear layer increases dimension from emb_size to d_ff
        self.linear1 = nn.Linear(emb_size, d_ff)  # W1 and B1

        # Dropout layer for regularization
        self.dropout = nn.Dropout(p=dropout)

        # Second linear layer reduces dimension back from d_ff to emb_size
        self.linear2 = nn.Linear(d_ff, emb_size)  # W2 and B2

    def forward(self, x):
        # Forward pass through the first linear layer and ReLU activation
        out = self.linear1(x)  # (Batch, seq_len, emb_size) -> (Batch, seq_len, d_ff)
        out = F.relu(out)  # Apply ReLU activation

        # Apply dropout for regularization
        out = self.dropout(out)

        # Forward pass through the second linear layer
        out = self.linear2(out)  # (Batch, seq_len, d_ff) -> (Batch, seq_len, emb_size)

        return out
