import torch
from torch import nn


class SimpleWordEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super(SimpleWordEmbedding, self).__init__()
        self.embeddings = nn.Parameter(torch.randn(vocab_size, embed_dim))
        self.linear = nn.Linear(embed_dim, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings[inputs]
        return self.linear(embeds)
