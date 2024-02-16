import logging

from torch import nn

from john_toolbox.train.transformers.network.decoder import Decoder, DecoderBlock
from john_toolbox.train.transformers.network.encoder import Encoder, EncoderBlock
from john_toolbox.train.transformers.network.layers.attention import MultiHeadAttentionBlock
from john_toolbox.train.transformers.network.layers.embeddings import (
    InputEmbeddings,
    PositionalEncoding,
)
from john_toolbox.train.transformers.network.layers.feed_forward import FeedForwardBlock
from john_toolbox.train.transformers.network.layers.projection import ProjectionLayer

LOGGER = logging.getLogger(__name__)


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
    tgt_vocab_size: int,
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
    tgt_emb = InputEmbeddings(emb_size, tgt_vocab_size)

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
    projection_layer = ProjectionLayer(emb_size=emb_size, vocab_size=tgt_vocab_size)

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
