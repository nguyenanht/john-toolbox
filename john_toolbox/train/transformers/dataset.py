import logging
from typing import Any

import torch
from torch.utils.data import Dataset

LOGGER = logging.getLogger(__name__)


class BilingualDataset(Dataset):
    """
    A Dataset class for handling bilingual data suitable for transformer architectures.
    It processes source and target language pairs for sequence-to-sequence tasks, such as
    machine translation. The class prepares the data by tokenizing, adding special tokens
    (SOS, EOS, PAD), and creating attention masks.

    Parameters
    ----------
    ds : Dataset
        The original dataset containing pairs of sentences in two languages.
    tokenizer_src : Tokenizer
        The tokenizer for the source language.
    tokenizer_tgt : Tokenizer
        The tokenizer for the target language.
    src_lang : str
        The source language code (e.g., 'en' for English).
    tgt_lang : str
        The target language code (e.g., 'fr' for French).
    seq_len : int
        The fixed sequence length for the model input. Longer sequences will be truncated,
        and shorter ones will be padded.

    Attributes
    ----------
    ds : Dataset
        Stores the original dataset.
    tokenizer_src : Tokenizer
        Tokenizer for the source language.
    tokenizer_tgt : Tokenizer
        Tokenizer for the target language.
    src_lang : str
        Source language code.
    tgt_lang : str
        Target language code.
    seq_len : int

    Methods
    -------
    __len__()
        Returns the length of the dataset.
    __getitem__(index)
        Returns a preprocessed item from the dataset at the specified index.

    Examples
    --------
    >>> from transformers import BertTokenizer
    >>> tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    >>> bilingual_data = BilingualDataset(ds, tokenizer, tokenizer, 'en', 'fr', 128)
    >>> print(bilingual_data[0])
    """

    def __init__(
        self,
        ds,
        tokenizer_src,
        tokenizer_tgt,
        src_lang,
        tgt_lang,
        seq_len,
    ) -> None:
        """
        Initializes the BilingualDataset object by setting up tokenizers and special tokens.

        Parameters are the same as described in the class documentation.
        """
        super().__init__()
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq_len = seq_len

        # start of sequence
        self.sos_token_id = torch.tensor([tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64)
        # end of sequence
        self.eos_token_id = torch.tensor([tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64)
        # padding token, sentances does not contains same number of word, we need to add pad tokens to have the same length
        self.pad_token_id = torch.tensor([tokenizer_tgt.token_to_id("[PAD]")], dtype=torch.int64)

    def __len__(self):
        """
        Returns the length of the dataset.

        Returns
        -------
        int
            The number of items in the dataset.
        """
        return len(self.ds)

    def __getitem__(self, index) -> Any:
        """
        Retrieves an item from the dataset and preprocesses it for transformer input.

        The method tokenizes the source and target sentences, adds special tokens (SOS, EOS),
        pads the sequences to a fixed length, and creates attention masks.

        Parameters
        ----------
        index : int
            The index of the item to be retrieved from the dataset.

        Returns
        -------
        dict
            A dictionary containing the preprocessed data for transformer input.

        Raises
        ------
        ValueError
            If either the source or target sentence is longer than `seq_len - 2`.
        """

        # Extract the source and target sentences from the dataset src
        src_target_pair = self.ds[index]
        src_text = src_target_pair["translation"][self.src_lang]
        tgt_text = src_target_pair["translation"][self.tgt_lang]

        # Tokenize the source and target sentences
        enc_input_tokens_ids = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens_ids = self.tokenizer_tgt.encode(tgt_text).ids

        # Calculate the number of padding tokens for the encoder input.
        # The '2' accounts for both the SOS and EOS tokens added to the sequence.
        # This ensures that the total length of the encoder input equals 'seq_len'.
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens_ids) - 2
        # Calculate the number of padding tokens for the decoder input.
        # The '1' accounts for the SOS token added to the start of the sequence.
        # The EOS token is not included in the decoder input, as it's used as a part of the output label.
        # This ensures that the total length of the decoder input equals 'seq_len'.
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens_ids) - 1

        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            LOGGER.error(len(enc_input_tokens_ids))
            LOGGER.error(len(dec_input_tokens_ids))
            LOGGER.error(self.seq_len)
            raise ValueError("Sentence too long.")

        # Concatenate SOS, EOS, and padding tokens with the tokenized source sentence
        encoder_input_ids = torch.cat(
            [
                self.sos_token_id,
                torch.tensor(enc_input_tokens_ids, dtype=torch.int64),
                self.eos_token_id,
                torch.tensor([self.pad_token_id] * enc_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        # Concatenate SOS and padding tokens with the tokenized target sentence for decoder input
        decoder_input_ids = torch.cat(
            [
                self.sos_token_id,
                torch.tensor(dec_input_tokens_ids, dtype=torch.int64),
                torch.tensor([self.pad_token_id] * dec_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )
        # `decoder_input_ids` are the input sequences provided to the decoder.
        # These sequences are temporally shifted versions of the target sequences
        # where each token is used to predict the next token in the sequence.

        # Temporal Shift Explanation:
        # The decoder_input_ids and the labels (targets) are shifted such that each
        # token in `decoder_input_ids` is aimed at predicting the subsequent token in the labels.
        # Adding an EOS (End of Sequence) token to `decoder_input_ids` would prematurely signal
        # to the model that sequence generation is complete, which is counterproductive for training
        # as the decoder is expected to predict the EOS token naturally, indicating the end of generation.

        # End-of-Sequence Learning:
        # Excluding the EOS token from `decoder_input_ids` but including it in the labels encourages
        # the model to autonomously learn to predict the EOS token. This approach trains the model
        # to recognize the end of a sequence organically, improving its ability to conclude sequence generation
        # without explicit prompts.

        # Consistency with Inference:
        # During inference or model testing, generation typically starts with just the SOS (Start of Sequence) token
        # and proceeds one token at a time until the EOS token is predicted, marking the sequence's end.
        # Omitting the EOS token from `decoder_input_ids` during training aligns the training behavior
        # closely with this inference process, fostering consistency across model operation stages.

        # Create the label for training by adding EOS and padding tokens to the target tokens
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens_ids, dtype=torch.int64),
                self.eos_token_id,
                torch.tensor([self.pad_token_id] * dec_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )
        # Ensure all tensors have the same sequence length as defined
        assert encoder_input_ids.size(0) == self.seq_len
        assert decoder_input_ids.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        # ! Create the encoder mask for attention mechanism.
        # This mask is a binary tensor indicating where the padding tokens are.
        # The mask has '1's where tokens are not padding and '0's where they are padding.
        # This allows the transformer's attention mechanism to ignore padding tokens.
        #
        # Example:
        # If the encoder input is [SOS, 15, 234, 67, EOS, PAD, PAD, PAD] (assuming 'seq_len' is 8),
        # then the encoder_mask would be [1, 1, 1, 1, 1, 0, 0, 0].
        # This indicates to the model that it should only attend to the first five tokens.
        encoder_mask = (encoder_input_ids != self.pad_token_id).unsqueeze(0).unsqueeze(0).int()

        # ! Create the decoder mask for the attention mechanism in the transformer.
        # This mask serves two purposes:
        # 1. It masks out padding tokens (similar to the encoder mask).
        # 2. It prevents the decoder from 'seeing' future tokens in the sequence,
        #    ensuring that the prediction for each token only depends on previous tokens.
        #
        # The mask is created by combining a padding mask and a causal mask.
        # Example:
        # If the decoder input is [SOS, 56, 78, 102, PAD, PAD, PAD, PAD] (assuming 'seq_len' is 8),
        # then the padding mask would be [1, 1, 1, 1, 0, 0, 0, 0].
        # The causal mask for this length would be a lower triangular matrix of size 8x8,
        # allowing each token to attend only to itself and preceding tokens.
        # The final decoder mask is the combination of these two masks.
        # tensor([[
        #  [1, 0, 0, 0, 0, 0, 0, 0],
        #  [1, 1, 0, 0, 0, 0, 0, 0],
        #  [1, 1, 1, 0, 0, 0, 0, 0],
        #  [1, 1, 1, 1, 0, 0, 0, 0],
        #  [1, 1, 1, 1, 0, 0, 0, 0],
        #  [1, 1, 1, 1, 0, 0, 0, 0],
        #  [1, 1, 1, 1, 0, 0, 0, 0],
        #  [1, 1, 1, 1, 0, 0, 0, 0]]
        # ], dtype=torch.int32)
        decoder_mask = (decoder_input_ids != self.pad_token_id).unsqueeze(0).unsqueeze(
            0
        ).int() & causal_mask(size=decoder_input_ids.size(0))

        return {
            "encoder_input": encoder_input_ids,  # (seq_len)
            "decoder_input": decoder_input_ids,  # (seq_len)
            "encoder_mask": encoder_mask,  # (1, 1, seq_len)
            "decoder_mask": decoder_mask,  # (1, seq_len) & (1, seq_len, seq_len)
            "label": label,  # (seq_len)
            "src_text": src_text,
            "tgt_text": tgt_text,
        }


def causal_mask(size):
    """
    Generates a causal mask to prevent attention to future tokens in a sequence.

    This function creates a mask for use in the self-attention mechanism of a transformer's
    decoder, ensuring that each position in the sequence can only attend to itself and
    positions before it. This maintains the autoregressive property of the decoder.

    Parameters
    ----------
    size : int
        The size of the sequence.

    Returns
    -------
    torch.Tensor
        A 2D tensor representing the causal mask, where the mask is 0 at and below the diagonal and 1 elsewhere.

    Example
    -------
    For a size of 4, the causal mask would be:
    [[0, 1, 1, 1],
     [0, 0, 1, 1],
     [0, 0, 0, 1],
     [0, 0, 0, 0]]

     The function uses `torch.triu` to create an upper triangular matrix, which is then inverted.
    """

    # Create an upper triangular matrix using torch.triu.
    # The 'diagonal=1' parameter sets the elements above the main diagonal to 1
    # (the main diagonal elements are set to 0).
    # This matrix represents which positions should initially be ignored (set to 1).
    # Example output for size=4:
    # [[0., 1., 1., 1.],
    #  [0., 0., 1., 1.],
    #  [0., 0., 0., 1.],
    #  [0., 0., 0., 0.]]
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int)

    # Invert the mask to create a causal mask.
    # After inversion, the positions set to 0 (current and past positions) are now 1,
    # allowing the model to attend to these positions.
    # The inverted (causal) mask for size=4 would be:
    # [[1, 0, 0, 0],
    #  [1, 1, 0, 0],
    #  [1, 1, 1, 0],
    #  [1, 1, 1, 1]]
    # Here, '1' indicates positions that are allowed to be attended to.
    return mask == 0
