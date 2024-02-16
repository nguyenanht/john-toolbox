import logging
import math

import torch
from torch import nn

LOGGER = logging.getLogger(__name__)


class InputEmbeddings(nn.Module):
    def __init__(self, emb_size: int, vocab_size: int) -> None:
        super().__init__()
        self.emb_size = emb_size
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, emb_size)

    def forward(self, x):
        # (batch, seq_len) --> (batch, seq_len, emb_size)
        # Multiply by sqrt(emb_size) to scale the embeddings according to the paper
        return self.embedding(x) * math.sqrt(self.emb_size)


class PositionalEncoding(nn.Module):
    """Implements the positional encoding as described in "Attention is All You Need".

    Positional encoding adds information about the position of each word in the input sequence
    to its corresponding embedding. This is important because the transformer architecture does
    not inherently process sequential data in order. The positional encoding uses a combination
    of sine and cosine functions with different frequencies to generate a unique encoding for
    each position up to a maximum sequence length.

    Attributes:
        emb_size (int):
            The embedding size, which is the dimensionality of the model's input.
            This is also referred to as `d_model` in the transformer literature.
        max_seq_len (int):
            The maximum length of the input sequences. This determines the size
            of the positional encoding matrix.
        dropout (nn.Dropout):
            Dropout layer to apply to the output of the positional encoding
            to prevent overfitting.
    """

    def __init__(self, emb_size: int, max_seq_len: int, dropout: float):
        """
        Initializes the PositionalEncoding module.

        Parameters:
            emb_size (int):
                The size of the embeddings (also called d_model in the transformer paper).
            max_seq_len (int):
                The maximum length of input sequences.
            dropout (float):
                The dropout rate to use after adding the positional encodings.
        """

        super().__init__()
        self.emb_size = emb_size
        self.max_seq_len = max_seq_len
        self.dropout = nn.Dropout(p=dropout)

        # ! create matrix of shape (max_seq_len, emb_size)
        # Initialize a positional encoding matrix with zeros.
        positional_encoding = torch.zeros(max_seq_len, emb_size)
        # example :
        # max_seq_len = 3
        # emb_size = 4

        # positional_encoding
        # tensor(
        # [[0., 0., 0., 0.],
        # [0., 0., 0., 0.],
        # [0., 0., 0., 0.]])

        # ! Create a vector of shape (max_seq_len, 1) : position of words inside a sentance
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        # tensor(
        # [[0.],
        # [1.],
        # [2.]]
        # )

        # ! Calculate the division term using the formula 2i/emb_size, where i is the dimension.
        # This is used to generate alternating sine and cosine waves with different frequencies.
        div_term = torch.exp(torch.arange(0, emb_size, 2).float() * (-math.log(10000.0) / emb_size))
        # tensor([1.0000, 0.0100])

        # ! Apply sine to even indices in the positional encoding matrix.
        positional_encoding[:, 0::2] = torch.sin(position * div_term)
        # tensor(
        #     [
        #         [0.0000, 0.0000, 0.0000, 0.0000],
        #         [0.8415, 0.0000, 0.0100, 0.0000],
        #         [0.9093, 0.0000, 0.0200, 0.0000]
        #     ]
        # )

        # ! Apply cosine to odd indices in the positional encoding matrix.
        positional_encoding[:, 1::2] = torch.cos(position * div_term)
        # tensor(
        #     [
        #         [ 0.0000,  1.0000,  0.0000,  1.0000],
        #         [ 0.8415,  0.5403,  0.0100,  0.9999],
        #         [ 0.9093, -0.4161,  0.0200,  0.9998]
        #     ]
        # )

        # ! Add a batch dimension to the positional encoding to enable broadcasting.
        positional_encoding = positional_encoding.unsqueeze(0)  # shape : (1, max_seq_len, emb_size)

        # ! Register positional_encoding as a buffer that is not a model parameter.
        # This is important because it should not be modified during training; it's fixed.
        self.register_buffer("positional_encoding", positional_encoding)

    def forward(self, x):
        # ! Add positional encoding to the input embedding, ensuring compatibility with the input sequence length.
        # x.shape : (batch, seq_len, emb_size)
        x = x + self.positional_encoding[:, : x.shape[1], :]  # type: ignore pylance
        # ! Apply dropout to the result for regularization.
        return self.dropout(x)
