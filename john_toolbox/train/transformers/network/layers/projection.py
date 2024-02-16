from torch import nn
import torch


class ProjectionLayer(nn.Module):
    """
    A ProjectionLayer in a transformer model for mapping the output of multi-head attention
    back to the vocabulary space.

    This layer takes the output from the multi-head attention mechanism, which is in the
    embedded space (emb_size), and projects it back into the vocabulary space. This is
    necessary for tasks like language modeling where the final output needs to be a
    probability distribution over the vocabulary.

    Parameters
    ----------
    emb_size : int
        The size of the embedding dimension. It's the size of the output from the previous
        layer (multi-head attention) in the transformer.

    vocab_size : int
        The size of the vocabulary. This is the dimensionality of the output space of this
        projection layer.

    Attributes
    ----------
    proj : nn.Linear
        A linear transformation that projects from the embedding space (emb_size) to the
        vocabulary space (vocab_size).

    Methods
    -------
    forward(x)
        Defines the forward pass of the ProjectionLayer. Applies a linear transformation
        and log softmax to the input tensor.

    """

    def __init__(self, emb_size: int, vocab_size: int):
        super().__init__()
        # Initialize a linear layer that projects from embedding dimension (emb_size) to vocabulary size (vocab_size)
        self.proj = nn.Linear(emb_size, vocab_size)

    def forward(self, x):
        # Apply the linear projection to the input
        # x shape: (Batch, seq_len, emb_size)
        # Output shape after projection: (Batch, seq_len, vocab_size)
        # The output is then passed through a log softmax function to convert it to log probabilities
        # Log softmax is often used for numerical stability compared to softmax
        return torch.log_softmax(input=self.proj(x), dim=-1)
