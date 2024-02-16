import logging
import warnings
from pathlib import Path

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer
from torch import nn
from torch.utils.data import DataLoader, random_split

from john_toolbox.train.transformers.config import (
    DATA_FOLDER,
    get_weights_file_path,
    latest_weights_file_path,
)
from john_toolbox.train.transformers.from_scratch.config import (
    get_config,
)
from john_toolbox.train.transformers.from_scratch.dataset import BilingualDataset
from john_toolbox.train.transformers.model import TransformerWrapper
from john_toolbox.utils.persist import create_local_folder

LOGGER = logging.getLogger(__name__)


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


if __name__ == "__main__":
    # config
    from john_toolbox.utils.logger_config import setup_log_config

    setup_log_config(is_dev=True, level="INFO")
    warnings.filterwarnings("ignore")
    config = get_config()
    create_local_folder(f"/work/data/{config['datasource']}_{config['model_folder']}")

    (
        train_dataloader,
        val_dataloader,
        tokenizer_src,
        tokenizer_tgt,
    ) = get_dataset(config)

    # launch training model

    model = TransformerWrapper(
        src_vocab_size=tokenizer_src.get_vocab_size(),
        tgt_vocab_size=tokenizer_tgt.get_vocab_size(),
        src_seq_len=config["seq_len"],
        tgt_seq_len=config["seq_len"],
        emb_size=config["d_model"],
        lr=config["lr"],
        device="auto",
        log_dir=f"/work/data/{config['experiment_name']}",
    )

    # If the user specified a model to preload before training, load it
    preload = config["preload"]
    model_filename = (
        latest_weights_file_path(
            datasource=config["datasource"],
            model_folder=config["model_folder"],
            model_basename=config["model_basename"],
        )
        if preload == "latest"
        else get_weights_file_path(
            datasource=config["datasource"],
            model_folder=config["model_folder"],
            model_basename=config["model_basename"],
            epoch=preload,
        )
        if preload
        else None
    )
    if model_filename:
        LOGGER.info(f"Preloading model {model_filename}")
        model.load_model(model_filename=model_filename)
    else:
        LOGGER.info("No model to preload, starting from scratch")

    # label_smoothing let the model be less confident to the decision.
    # in other words, if the model give a very high probability to a next token,
    # it will distribute a small amount of the probability to the other tokens
    # so that the model become less sure of its choices (avoid overfitting)
    loss_fn = nn.CrossEntropyLoss(
        ignore_index=tokenizer_src.token_to_id("[PAD]"), label_smoothing=0.1
    )
    save_path = (
        f"{DATA_FOLDER}/{config['datasource']}_{config['model_folder']}/{config['model_basename']}"
    )

    model.train(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        tokenizer_src=tokenizer_src,
        tokenizer_tgt=tokenizer_tgt,
        max_seq_len=config["seq_len"],
        lr=config["lr"],
        loss_fn=loss_fn,
        epochs=config["num_epochs"],
        save_path=save_path,
    )
