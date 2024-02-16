import logging
import os
import warnings
from pathlib import Path

import torch
import torch.nn as nn
import torchmetrics
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard.writer import SummaryWriter

from john_toolbox.train.transformers.from_scratch.config import (
    get_config,
    get_weights_file_path,
    latest_weights_file_path,
)
from john_toolbox.train.transformers.from_scratch.dataset import BilingualDataset, causal_mask
from john_toolbox.train.transformers.from_scratch.model import build_transformer
from john_toolbox.utils.helper import get_tqdm_func
from john_toolbox.utils.persist import create_local_folder

LOGGER = logging.getLogger(__name__)


def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.token_to_id("[SOS]")
    eos_idx = tokenizer_tgt.token_to_id("[EOS]")

    # Precompute the encoder output and reuse it for every step
    encoder_output = model.encode(source, source_mask)
    # Initialize the decoder input with the sos token
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    while True:
        if decoder_input.size(1) == max_len:
            break

        # build mask for target
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        # calculate output
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        # get next token
        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)],
            dim=1,
        )

        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)


def run_validation(
    model,
    validation_ds,
    tokenizer_src,
    tokenizer_tgt,
    max_len,
    device,
    print_msg,
    global_step,
    writer,
    num_examples=2,
):
    model.eval()
    count = 0

    source_texts = []
    expected = []
    predicted = []

    try:
        # get the console window width
        with os.popen("stty size", "r") as console:
            _, console_width = console.read().split()
            console_width = int(console_width)
    except Exception:
        # If we can't get the console width, use 80 as default
        console_width = 80

    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch["encoder_input"].to(device)  # (b, seq_len)
            encoder_mask = batch["encoder_mask"].to(device)  # (b, 1, 1, seq_len)

            # check that the batch size is 1
            assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"

            model_out = greedy_decode(
                model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device
            )

            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)

            # Print the source, target and model output
            print_msg("-" * console_width)
            print_msg(f"{f'SOURCE: ':>12}{source_text}")
            print_msg(f"{f'TARGET: ':>12}{target_text}")
            print_msg(f"{f'PREDICTED: ':>12}{model_out_text}")

            if count == num_examples:
                print_msg("-" * console_width)
                break

    if writer:
        # Evaluate the character error rate
        # Compute the char error rate
        metric = torchmetrics.CharErrorRate()
        cer = metric(predicted, expected)
        writer.add_scalar("validation cer", cer, global_step)
        writer.flush()

        # Compute the word error rate
        metric = torchmetrics.WordErrorRate()
        wer = metric(predicted, expected)
        writer.add_scalar("validation wer", wer, global_step)
        writer.flush()

        # Compute the BLEU metric
        metric = torchmetrics.BLEUScore()
        bleu = metric(predicted, expected)
        writer.add_scalar("validation BLEU", bleu, global_step)
        writer.flush()


def get_all_sentences(ds, lang):
    for item in ds:
        yield item["translation"][lang]


def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path = Path(f'/work/data/{config["tokenizer_file"].format(lang)}')
    if not Path.exists(tokenizer_path):
        # Most code taken from: https://huggingface.co/docs/tokenizers/quicktour
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(
            special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2
        )
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer


def get_dataset(config):
    # https://huggingface.co/datasets/opus_books/viewer/en-fr
    ds_raw = load_dataset(
        f"{config['datasource']}",
        f"{config['lang_src']}-{config['lang_tgt']}",
        split="train",
    )

    # build tokenizers
    tokenizer_src = get_or_build_tokenizer(config=config, ds=ds_raw, lang=config["lang_src"])
    tokenizer_tgt = get_or_build_tokenizer(config=config, ds=ds_raw, lang=config["lang_tgt"])

    # keep 90% for training and 10% for validation
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    train_ds = BilingualDataset(
        ds=train_ds_raw,
        tokenizer_src=tokenizer_src,
        tokenizer_tgt=tokenizer_tgt,
        src_lang=config["lang_src"],
        tgt_lang=config["lang_tgt"],
        seq_len=config["seq_len"],
    )

    val_ds = BilingualDataset(
        ds=val_ds_raw,
        tokenizer_src=tokenizer_src,
        tokenizer_tgt=tokenizer_tgt,
        src_lang=config["lang_src"],
        tgt_lang=config["lang_tgt"],
        seq_len=config["seq_len"],
    )

    max_len_src = 0
    max_len_tgt = 0

    for item in ds_raw:
        src_ids = tokenizer_src.encode(item["translation"][config["lang_src"]]).ids
        tgt_ids = tokenizer_tgt.encode(item["translation"][config["lang_tgt"]]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    LOGGER.info(f"Max length of source sentence : {max_len_src}")
    LOGGER.info(f"Max length of target sentence : {max_len_tgt}")

    train_dataloader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt


def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(
        src_vocab_size=vocab_src_len,
        tgt_vocab=vocab_tgt_len,
        src_seq_len=config["seq_len"],
        tgt_seq_len=config["seq_len"],
        emb_size=config["d_model"],
    )
    return model


def train_model(config):
    # Define the device
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.has_mps or torch.backends.mps.is_available()
        else "cpu"
    )
    LOGGER.info(
        f"Using device: {device}",
    )
    if device == "cuda":
        LOGGER.info(f"Device name: {torch.cuda.get_device_name(device.index)}")
        LOGGER.info(
            f"Device memory: {torch.cuda.get_device_properties(device.index).total_memory / 1024 ** 3} GB"
        )
    elif device == "mps":
        LOGGER.info("Device name: <mps>")
    else:
        LOGGER.info("NOTE: If you have a GPU, consider using it for training.")
        LOGGER.info(
            "      On a Windows machine with NVidia GPU, check this video: https://www.youtube.com/watch?v=GMSjDTU8Zlc"
        )
        LOGGER.info(
            "      On a Mac machine, run: pip3 install --pre torch torchvision torchaudio torchtext --index-url https://download.pytorch.org/whl/nightly/cpu"
        )
    device = torch.device(device)

    # Make sure the weights folder exists
    create_local_folder(f"/work/data/{config['datasource']}_{config['model_folder']}")

    (
        train_dataloader,
        val_dataloader,
        tokenizer_src,
        tokenizer_tgt,
    ) = get_dataset(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(
        device
    )
    # Tensorboard
    writer = SummaryWriter(log_dir=f"/work/data/{config['experiment_name']}")

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], eps=1e-9)

    # If the user specified a model to preload before training, load it
    initial_epoch = 0
    global_step = 0
    preload = config["preload"]
    model_filename = (
        latest_weights_file_path(config)
        if preload == "latest"
        else get_weights_file_path(config, preload)
        if preload
        else None
    )
    if model_filename:
        LOGGER.info(f"Preloading model {model_filename}")
        state = torch.load(model_filename)
        model.load_state_dict(state["model_state_dict"])
        initial_epoch = state["epoch"] + 1
        optimizer.load_state_dict(state["optimizer_state_dict"])
        global_step = state["global_step"]
    else:
        LOGGER.info("No model to preload, starting from scratch")

    # label_smoothing let the model be less confident to the decision.
    # in other words, if the model give a very high probability to a next token,
    # it will distribute a small amount of the probability to the other tokens
    # so that the model become less sure of its choices (avoid overfitting)
    loss_fn = nn.CrossEntropyLoss(
        ignore_index=tokenizer_src.token_to_id("[PAD]"), label_smoothing=0.1
    ).to(device)
    tqdm_func = get_tqdm_func()

    for epoch in range(initial_epoch, config["num_epochs"]):
        torch.cuda.empty_cache()
        model.train()

        batch_iterator = tqdm_func(train_dataloader, desc=f"Processing Epoch {epoch:02d}")
        for batch in batch_iterator:
            # LOGGER.error("ici")
            encoder_input = batch["encoder_input"].to(device)  # (b, seq_len)
            # LOGGER.debug(encoder_input.shape)
            decoder_input = batch["decoder_input"].to(device)  # (B, seq_len)
            # LOGGER.debug(decoder_input.shape)

            encoder_mask = batch["encoder_mask"].to(
                device
            )  # (B, 1, 1, seq_len) it hides only the padding tokens
            # LOGGER.debug(encoder_mask.shape)
            decoder_mask = batch["decoder_mask"].to(
                device
            )  # (B, 1, seq_len, seq_len) it hides the padding tokens AND all the subsequent words

            # Run the tensors through the encoder, decoder and the projection layer
            encoder_output = model.encode(encoder_input, encoder_mask)  # (B, seq_len, d_model)
            # LOGGER.debug(encoder_output.shape)

            decoder_output = model.decode(
                encoder_output, encoder_mask, decoder_input, decoder_mask
            )  # (B, seq_len, d_model)
            # LOGGER.debug(decoder_output.shape)

            proj_output = model.project(decoder_output)  # (B, seq_len, vocab_size)
            # LOGGER.debug(proj_output.shape)

            # Compare the output with the label
            label = batch["label"].to(device)  # (B, seq_len)

            # Compute the loss using a simple cross entropy
            # B, seq_len, tgt_vocab_size) --> (B, seq_len tgt_vocab_size)
            loss = loss_fn(
                proj_output.view(-1, tokenizer_tgt.get_vocab_size()),
                label.view(-1),
            )
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

            # Log the loss
            writer.add_scalar("train loss", loss.item(), global_step)
            writer.flush()

            # Backpropagate the loss
            loss.backward()

            # Update the weights
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1
            pass
            # LOGGER.error("end of one batch")

        # Run validation at the end of every epoch
        run_validation(
            model,
            val_dataloader,
            tokenizer_src,
            tokenizer_tgt,
            config["seq_len"],
            device,
            lambda msg: batch_iterator.write(msg),
            global_step,
            writer,
        )

        # Save the model at the end of every epoch
        model_filename = get_weights_file_path(config=config, epoch=f"{epoch:02d}")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "global_step": global_step,
            },
            model_filename,
        )
        LOGGER.error("end of one epoch")
        pass


if __name__ == "__main__":
    from john_toolbox.utils.logger_config import setup_log_config

    setup_log_config(is_dev=True, level="INFO")
    warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)
