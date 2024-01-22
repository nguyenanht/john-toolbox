import logging
import math

import torch
import torch.nn.functional as F
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
    """docstring for PositionalEncoding.

    A word must be in the same position in the embedding space independantly to the sentence input lenght.
    To do this, we introduce the wave frequencies concept here.

    """

    def __init__(self, emb_size: int, max_seq_len: int, dropout: float):
        """_summary_

        Parameters
        ----------
        emb_size : int
            the size of the position encoding should be, it is the same as the size the embedding.
            it is also called d_model in the paper Attention is all you need.
        max_seq_len : int
            maximum length of the sentence. We need to create one vector for each position.
        dropout : float
            to prevent overfitting.
        """

        super(PositionalEncoding, self).__init__()
        self.emb_size = emb_size
        self.max_seq_len = max_seq_len
        self.dropout = nn.Dropout(p=dropout)

        # ! create matrix of shape (max_seq_len, emb_size)
        positional_encoding = torch.zeros(max_seq_len, emb_size)
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

        div_term = torch.exp(torch.arange(0, emb_size, 2).float() * (-math.log(10000.0) / emb_size))
        # tensor([1.0000, 0.0100])

        positional_encoding[:, 0::2] = torch.sin(position * div_term)
        # tensor(
        #     [
        #         [0.0000, 0.0000, 0.0000, 0.0000],
        #         [0.8415, 0.0000, 0.0100, 0.0000],
        #         [0.9093, 0.0000, 0.0200, 0.0000]
        #     ]
        # )

        positional_encoding[:, 1::2] = torch.cos(position * div_term)
        # tensor(
        #     [
        #         [ 0.0000,  1.0000,  0.0000,  1.0000],
        #         [ 0.8415,  0.5403,  0.0100,  0.9999],
        #         [ 0.9093, -0.4161,  0.0200,  0.9998]
        #     ]
        # )

        positional_encoding = positional_encoding.unsqueeze(0)  # shape : (1, max_seq_len, emb_size)

        # we keep the positional encoding in the module, not as a parameter that will be learn
        self.register_buffer("positional_encoding", positional_encoding)

    def forward(self, x):
        # x.shape : (batch, seq_len, emb_size)
        x = x + self.positional_encoding[:, : x.shape[1], :]  # type: ignore pylance
        return self.dropout(x)


class LayerNormalization(nn.Module):
    """We normalize data in the features axis and not on the batch to help training."""

    def __init__(
        self,
        features: int,
        eps: float = 10**-6,
    ):
        super(LayerNormalization, self).__init__()
        self.eps = eps  # "Epsilon helps avoid problems when sigma is too large or equal to 0."
        self.alpha = nn.Parameter(data=torch.ones(features))  # Multiplied
        self.bias = nn.Parameter(data=torch.zeros(features))  # Added

    def forward(self, x):
        # x: (batch, seq_len, hidden_size)
        # Keep the dimension for broadcasting
        # (batch, seq_len, 1)
        mean = x.mean(dim=-1, keepdim=True)
        # Keep the dimension for broadcasting
        # (batch, seq_len, 1)
        std = x.std(dim=-1, keepdim=True)
        # eps is to prevent dividing by zero or when std is very small
        return self.alpha * (x - mean) / (std + self.eps) + self.bias


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
        super(FeedForwardBlock, self).__init__()
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
        super(MultiHeadAttentionBlock, self).__init__()
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
        # the transpose is imoportant because we want each head to to watch this (seq_len, d_k)
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


class EncoderBlock(nn.Module):
    """docstring for EncoderBlock."""

    def __init__(
        self,
        features: int,
        self_attention_block: MultiHeadAttentionBlock,
        feed_forward_block: FeedForwardBlock,
        dropout: float,
    ):
        super(EncoderBlock, self).__init__()
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
        super(Encoder, self).__init__()
        self.layers = layers
        self.norm = LayerNormalization(features=features)

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        for layer in self.layers:
            x = layer(x, mask)

        return self.norm(x)


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
        super(DecoderBlock, self).__init__()

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
        super(Decoder, self).__init__()
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


class Transformer(nn.Module):
    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        src_emb: InputEmbeddings,
        tgt_emb: InputEmbeddings,
        src_pos: PositionalEncoding,
        tgt_pos: PositionalEncoding,
        projection_layer: ProjectionLayer,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_emb = src_emb
        self.tgt_emb = tgt_emb
        self.src_positional_encoding = src_pos
        self.tgt_positional_encoding = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        src = self.src_emb(src)
        src = self.src_positional_encoding(src)

        return self.encoder(src, src_mask)

    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        tgt = self.tgt_emb(tgt)
        tgt = self.tgt_positional_encoding(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

    def project(self, x):
        if x is None:
            raise ValueError("x cannot be None")
        return self.projection_layer(x)


def build_transformer(
    src_vocab_size: int,
    tgt_vocab: int,
    src_seq_len: int,
    tgt_seq_len: int,
    emb_size: int = 512,
    N: int = 6,
    h: int = 8,
    dropout: float = 0.1,
    d_ff: int = 2048,
) -> Transformer:
    """
    Constructs a Transformer model with specified hyperparameters.

    Parameters
    ----------
    src_vocab_size : int
        Size of the source vocabulary.
    tgt_vocab : int
        Size of the target vocabulary.
    src_seq_len : int
        Maximum sequence length for the source text.
    tgt_seq_len : int
        Maximum sequence length for the target text.
    emb_size : int, optional
        Size of the embedding space (default is 512).
    N : int, optional
        Number of layers in both the encoder and decoder (default is 6).
    h : int, optional
        Number of heads in the multi-head attention mechanism (default is 8).
    dropout : float, optional
        Dropout rate for regularization (default is 0.1).
    d_ff : int, optional
        Dimension of the feedforward network model (default is 2048).

    Returns
    -------
    Transformer
        An instance of the Transformer model.

    """

    # Create embedding layers for source and target
    src_emb = InputEmbeddings(emb_size, src_vocab_size)
    tgt_emb = InputEmbeddings(emb_size, tgt_vocab)

    # Create positional encoding layers for source and target
    src_pos = PositionalEncoding(emb_size=emb_size, max_seq_len=src_seq_len, dropout=dropout)
    tgt_pos = PositionalEncoding(emb_size=emb_size, max_seq_len=tgt_seq_len, dropout=dropout)

    features = emb_size
    # features = 1

    # Create N encoder blocks, each comprising a multi-head attention and a feed-forward network
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(
            emb_size=emb_size, h=h, dropout=dropout
        )
        feed_forward_block = FeedForwardBlock(emb_size=emb_size, d_ff=d_ff, dropout=dropout)
        encoder_block = EncoderBlock(
            features=features,
            self_attention_block=encoder_self_attention_block,
            feed_forward_block=feed_forward_block,
            dropout=dropout,
        )
        encoder_blocks.append(encoder_block)

    # Create N decoder blocks, each comprising two multi-head attentions and a feed-forward network
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(
            emb_size=emb_size, h=h, dropout=dropout
        )
        decoder_cross_attention_block = MultiHeadAttentionBlock(
            emb_size=emb_size, h=h, dropout=dropout
        )
        feed_forward_block = FeedForwardBlock(emb_size=emb_size, d_ff=d_ff, dropout=dropout)
        decoder_block = DecoderBlock(
            features=features,
            self_attention_block=decoder_self_attention_block,
            cross_attention_block=decoder_cross_attention_block,
            feed_forward_block=feed_forward_block,
            dropout=dropout,
        )
        decoder_blocks.append(decoder_block)

    # Create the encoder and decoder from the assembled blocks
    encoder = Encoder(features=features, layers=nn.ModuleList(encoder_blocks))
    decoder = Decoder(features=features, layers=nn.ModuleList(decoder_blocks))

    # Create the projection layer to map the decoder output to the target vocabulary space
    projection_layer = ProjectionLayer(emb_size=emb_size, vocab_size=tgt_vocab)

    # Assemble the transformer model
    transformer = Transformer(
        encoder=encoder,
        decoder=decoder,
        src_emb=src_emb,
        tgt_emb=tgt_emb,
        src_pos=src_pos,
        tgt_pos=tgt_pos,
        projection_layer=projection_layer,
    )

    # Initialize parameters using the Xavier uniform initialization for better convergence
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer
