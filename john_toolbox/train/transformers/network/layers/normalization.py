from torch import nn
import torch


class LayerNormalization(nn.Module):
    """
    Implements Layer Normalization as described in the paper "Layer Normalization".

    Layer Normalization normalizes the input across the features instead of normalizing across the batch dimension
    as in Batch Normalization. This normalization technique is particularly useful in stabilizing the hidden state
    dynamics in recurrent and transformer models. It computes the mean and standard deviation used for normalization
    from all of the summed inputs to the neurons in a layer on a single training case.

    Unlike Batch Normalization, Layer Normalization performs the same computation at training and test times.

    Attributes:
        features (int):
            The number of features in the input tensor. For a transformer, this would be the embedding size.
        eps (float):
            A small epsilon value added to the denominator for numerical stability. Defaults to 1e-6.
        alpha (nn.Parameter):
            A learnable scale factor applied after normalization. Initialized to ones.
        bias (nn.Parameter):
            A learnable bias term applied after normalization. Initialized to zeros.
    """

    def __init__(
        self,
        features: int,
        eps: float = 10**-6,
    ):
        """
        Initializes the LayerNormalization module.

        Parameters:
            features (int):
                The number of features in the input tensor.
            eps (float):
                A small epsilon value for numerical stability in division.
        """
        super().__init__()
        self.eps = eps  # "Epsilon helps avoid problems when sigma is too large or equal to 0."
        self.alpha = nn.Parameter(data=torch.ones(features))  # Multiplied
        self.bias = nn.Parameter(data=torch.zeros(features))  # Added

    def forward(self, x):
        # ! Compute the mean and standard deviation on the features dimension.
        # x: (batch, seq_len, hidden_size)
        # Keep the dimension for broadcasting
        # (batch, seq_len, 1)
        mean = x.mean(dim=-1, keepdim=True)
        # Keep the dimension for broadcasting
        # (batch, seq_len, 1)
        std = x.std(dim=-1, keepdim=True)
        # eps is to prevent dividing by zero or when std is very small

        # Normalize the input tensor using computed mean and std, then scale and shift using learnable parameters.
        # This normalization ensures each feature has zero mean and unit variance, then is scaled and shifted to potentially
        # regain the original representation if needed by the model.
        normalized_x = (x - mean) / (std + self.eps)  # Adding eps improves numerical stability.

        return (
            self.alpha * normalized_x + self.bias
        )  # Scale (alpha) and shift (bias) the normalized output.
