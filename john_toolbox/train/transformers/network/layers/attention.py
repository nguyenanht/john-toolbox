import logging
import math

from torch import nn

LOGGER = logging.getLogger(__name__)


class MultiHeadAttentionBlock(nn.Module):
    """
    MultiHeadAttentionBlock implements the multi-head attention mechanism from the Transformer model.

    This block allows the model to jointly attend to information from different representation
    subspaces at different positions. Each head in the multi-head attention mechanism splits
    the input embedding into smaller chunks and processes them independently. This enables
    the model to capture different aspects of the information contained in the input.

    The process involves linearly projecting the queries, keys, and values h times with
    different, learned linear projections. Each projected version of the queries, keys,
    and values are then passed through the attention function in parallel, producing
    dv-dimensional output values. These are concatenated and once again projected, resulting
    in the final values.

    In the input of the multi-head attention, we have an embedding matrix of seq_len x 512-d embedding vectors,
    where seq_len is the number of tokens in the sentence.
    We divide each token into 8 (number of head) chunks. This results in -> 512 / 8 = 64.

    Example:
    In the sentence 'Anthony Hopkins admired Michael Bay as a great director',
    we have 9 tokens in the input:
    input_embedding = 9 x 512

    Each head attention takes as input one of the 8 chunks of the embedding:
    Q, K, V = 9 x 64

    Parameters
    ----------
    emb_size : int
        The dimension of the input embeddings.
    h : int
        The number of attention heads.
    dropout : float
        Dropout rate to prevent overfitting.

    Attributes
    ----------
    emb_size : int
        The size of the input embeddings.
    h : int
        The number of heads in the multi-head attention mechanism.
    d_k : int
        The size of each attention head.
    w_q, w_k, w_v : nn.Linear
        Linear layers for queries, keys, and values.
    w_o : nn.Linear
        Output linear layer to concatenate heads' outputs.
    dropout : nn.Dropout
        Dropout layer for regularization.
    """

    def __init__(self, emb_size: int, h: int, dropout: float):
        """

        Parameters
        ----------
        emb_size : int
            the dimension of the input
        h : int
            number of head, in the paper it is 8
        dropout : float
            To prevent overfitting
        """
        super().__init__()
        self.emb_size = emb_size
        self.h = h
        assert emb_size % h == 0, "emb_size is not divisible by h the number of head."
        self.dropout = nn.Dropout(p=dropout)

        self.d_k = emb_size // h  # 512 // 8 = 64, each head will handle an input of size self.d_k

        self.w_q = nn.Linear(self.emb_size, self.emb_size)  # wq
        self.w_k = nn.Linear(self.emb_size, self.emb_size)  # wk
        self.w_v = nn.Linear(self.emb_size, self.emb_size)  # wv
        # in the paper dv is dk because of the result of the attention(Q,K,V)
        self.d_v = self.d_k
        self.w_o = nn.Linear(h * self.d_v, self.emb_size)  # wo
        self.dropout = nn.Dropout(p=dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        # (batch, h, seq_len, d_k)

        # Extract the dimension 'd_k' from the last dimension of 'query'. This represents the size of the key vectors.
        d_k = query.shape[-1]

        # Transpose the 'key' tensor to swap the last two dimensions.
        # This is done to align the dimensions for the matrix multiplication with 'query'.
        # The transpose changes the shape from (batch, heads, seq_len, d_k) to (batch, heads, d_k, seq_len).
        key_transposed = key.transpose(-2, -1)

        # Compute the attention scores.
        # The operation 'query @ key_transposed' is a batched matrix multiplication.
        # 'math.sqrt(d_k)' is used to scale the scores, which helps with stability and performance.
        # (batch, h, seq_len, d_k) -> (batch, h, seq_len, seq_len)
        attention_scores = (query @ key_transposed) / math.sqrt(d_k)

        # Apply the mask, if it's provided.
        # The mask is applied to the attention scores, setting certain positions to a very small value (-1e9).
        # This effectively removes these positions from consideration in the subsequent softmax operation.
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)  # the softmax will replace by 0

        # Apply softmax to normalize the attention scores.
        # 'dim=-1' means softmax is applied along the last dimension, which in this case corresponds to the sequence length.
        # This ensures that the attention scores for each position sum to 1, turning them into probabilities.
        # (batch, seq_len, seq_len)
        attention_scores = attention_scores.softmax(dim=-1)
        # The dim=-1 parameter in the softmax function is critical.
        # It specifies that the softmax should be applied across
        # the last dimension of the input tensor (attention_scores).
        # In the context of attention scores, this last dimension typically
        # corresponds to the sequence length in NLP tasks,
        # meaning each set of scores corresponding
        # to a single input position is normalized to sum up to 1,
        # (batch, h, seq_len, seq_len) --> (batch, h, seq_len, d_k)
        # which is essential for them to be used as weights in the attention mechanism.
        if dropout:
            attention_scores = dropout(attention_scores)
        return (
            attention_scores @ value
        ), attention_scores  # second part of the tuple is only for visualization

    def forward(self, q, k, v, mask):
        """_summary_

        Parameters
        ----------
        q : _type_
            _description_
        k : _type_
            _description_
        v : _type_
            _description_
        mask : _type_
            The mask is basically if we want some words to not interract with some other words, we mask them
        """
        # input : (seq_len, emb_size) -> Q, K, V (seq_len, emb_size)
        # (batch, seq_len, emb_size= -> ( batch, seq_len, emb_size)
        query = self.w_q(q)
        # (batch, seq_len, emb_size= -> ( batch, seq_len, emb_size)
        key = self.w_k(k)
        # (batch, seq_len, emb_size= -> ( batch, seq_len, emb_size)
        value = self.w_v(v)

        # (Batch, seq_len, emb_size) -> (batch, seq_len, h, d_k) --> (batch, h, seq_len, d_k)
        # the transpose is important because we want each head to to watch this (seq_len, d_k)
        # in other words, each head see all the words but a specific part of the embeddings
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        x, self.attention_scores = MultiHeadAttentionBlock.attention(
            query, key, value, mask, self.dropout
        )
        # (batch, h, seq_len, d_k) -> (batch, seq_len, h, d_k) -> (batch, seq_len, emb_size)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        # (batch, seq_len, emb_size} -> (batch, seq_len, emb_size)
        return self.w_o(x)
