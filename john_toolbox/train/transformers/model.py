import logging
import os

import torch
import torchmetrics
from torch.utils.tensorboard.writer import SummaryWriter

from john_toolbox.train.transformers.dataset import causal_mask
from john_toolbox.train.transformers.network.transformers import build_transformer
from john_toolbox.utils.helper import get_tqdm_func

LOGGER = logging.getLogger(__name__)


class TransformerWrapper:
    """
    Wrapper pour le modèle de transformateur, gérant l'initialisation, l'entraînement et la prédiction.

    Paramètres pour l'initialisation du modèle Transformer:
        num_layers (int): Nombre de couches dans l'encodeur et le décodeur.
        emb_size (int): Taille de l'embedding.
        num_heads (int): Nombre de têtes dans l'attention multi-têtes.
        ff_size (int): Taille de la couche feedforward interne.
        src_vocab_size (int): Taille du vocabulaire source.
        tgt_vocab_size (int): Taille du vocabulaire cible.
        max_length (int): Longueur maximale des séquences.
        dropout (float): Taux de dropout pour la régularisation.
        learning_rate (float): Taux d'apprentissage pour l'optimiseur.
    """

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        src_seq_len: int,
        tgt_seq_len: int,
        emb_size: int,
        log_dir: str,
        device="auto",
    ):
        # Define the device
        self.device = self._set_device(device=device)
        self.model = build_transformer(
            src_vocab_size=src_vocab_size,
            tgt_vocab_size=tgt_vocab_size,
            src_seq_len=src_seq_len,
            tgt_seq_len=tgt_seq_len,
            emb_size=emb_size,
        ).to(self.device)

        # Tensorboard
        self.writer = SummaryWriter(log_dir=log_dir)

        # training attributes
        self.initial_epoch = 0
        self.global_step = 0

    def _set_device(self, device: str):
        if device.lower() == "auto":
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

        return torch.device(device)

    def train(
        self,
        max_seq_len,
        train_dataloader,
        val_dataloader,
        tokenizer_src,
        tokenizer_tgt,
        loss_fn,
        save_path,
        lr: float,
        epochs=1,
    ):
        """
        Entraîne le modèle Transformer.

        Paramètres:
            src (Tensor): Tenseur de la séquence source.
            tgt (Tensor): Tenseur de la séquence cible.
            src_mask (Tensor): Masque pour la séquence source.
            tgt_mask (Tensor): Masque pour la séquence cible.
            epochs (int): Nombre d'époques pour l'entraînement.
        """
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, eps=1e-9)
        self.loss_fn = loss_fn

        tqdm_func = get_tqdm_func()
        # TODO : rework the range, it is weired if we does not train from scratch
        # initial_epoch = 0
        # global_step = 0

        for epoch in range(self.initial_epoch, epochs):
            torch.cuda.empty_cache()
            self.model.train()
            batch_iterator = tqdm_func(train_dataloader, desc=f"Processing Epoch {epoch:02d}")
            for batch in batch_iterator:
                # LOGGER.error("ici")
                encoder_input = batch["encoder_input"].to(self.device)  # (b, seq_len)
                # LOGGER.debug(encoder_input.shape)
                decoder_input = batch["decoder_input"].to(self.device)  # (B, seq_len)
                # LOGGER.debug(decoder_input.shape)

                encoder_mask = batch["encoder_mask"].to(
                    self.device
                )  # (B, 1, 1, seq_len) it hides only the padding tokens
                # LOGGER.debug(encoder_mask.shape)
                decoder_mask = batch[
                    "decoder_mask"
                ].to(
                    self.device
                )  # (B, 1, seq_len, seq_len) it hides the padding tokens AND all the subsequent words

                # Run the tensors through the encoder, decoder and the projection layer
                encoder_output = self.model.encode(
                    encoder_input, encoder_mask
                )  # (B, seq_len, d_model)
                # LOGGER.debug(encoder_output.shape)

                decoder_output = self.model.decode(
                    encoder_output, encoder_mask, decoder_input, decoder_mask
                )  # (B, seq_len, d_model)
                # LOGGER.debug(decoder_output.shape)

                proj_output = self.model.project(decoder_output)  # (B, seq_len, vocab_size)
                # LOGGER.debug(proj_output.shape)

                # Compare the output with the label
                label = batch["label"].to(self.device)  # (B, seq_len)

                # Compute the loss using a simple cross entropy
                # B, seq_len, tgt_vocab_size) --> (B, seq_len tgt_vocab_size)
                loss = self.loss_fn(
                    proj_output.view(-1, tokenizer_tgt.get_vocab_size()),
                    label.view(-1),
                )
                batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

                # Log the loss
                self.writer.add_scalar("train loss", loss.item(), self.global_step)
                self.writer.flush()

                # Backpropagate the loss
                loss.backward()

                # Update the weights
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

                self.global_step += 1
                pass
            # Run validation at the end of every epoch
            self.run_validation(
                validation_ds=val_dataloader,
                tokenizer_src=tokenizer_src,
                tokenizer_tgt=tokenizer_tgt,
                max_len=max_seq_len,
                print_msg=lambda msg: batch_iterator.write(msg),  # noqa
            )

            # Save the model at the end of every epoch
            model_filename = f"{save_path}{epoch:02d}.pt"

            if save_path is not None:
                self._save_model_epoch(
                    epoch=epoch,
                    model_state_dict=self.model.state_dict(),
                    optimizer_state_dict=self.optimizer.state_dict(),
                    global_step=self.global_step,
                    model_filename=model_filename,
                )
            LOGGER.error("end of one epoch")
            pass
        pass

    def predict(self, src, src_mask):
        """
        Fait une prédiction en utilisant le modèle Transformer.

        Paramètres:
            src (Tensor): Tenseur de la séquence source pour la prédiction.
            src_mask (Tensor): Masque pour la séquence source.

        Retourne:
            Tensor: Prédictions du modèle.
        """
        self.model.eval()
        with torch.no_grad():
            preds = self.model(src, src, src_mask, src_mask)
            return preds

    def load_model(self, model_filename):
        state = torch.load(model_filename)
        self.model.load_state_dict(state["model_state_dict"])
        self.initial_epoch = state["epoch"] + 1
        self.optimizer.load_state_dict(state["optimizer_state_dict"])
        self.global_step = state["global_step"]
        pass

    def _save_model_epoch(
        self,
        epoch,
        model_state_dict,
        optimizer_state_dict,
        global_step,
        model_filename,
    ):
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model_state_dict,
                "optimizer_state_dict": optimizer_state_dict,
                "global_step": global_step,
            },
            model_filename,
        )
        LOGGER.error("end of one epoch")

    def greedy_decode(self, source, source_mask, tokenizer_src, tokenizer_tgt, max_len):
        sos_idx = tokenizer_tgt.token_to_id("[SOS]")
        eos_idx = tokenizer_tgt.token_to_id("[EOS]")

        # Precompute the encoder output and reuse it for every step
        encoder_output = self.model.encode(source, source_mask)
        # Initialize the decoder input with the sos token
        decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(self.device)
        while True:
            if decoder_input.size(1) == max_len:
                break

            # build mask for target
            decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(self.device)

            # calculate output
            out = self.model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

            # get next token
            prob = self.model.project(out[:, -1])
            _, next_word = torch.max(prob, dim=1)
            decoder_input = torch.cat(
                [
                    decoder_input,
                    torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(self.device),
                ],
                dim=1,
            )

            if next_word == eos_idx:
                break

        return decoder_input.squeeze(0)

    def run_validation(
        self,
        validation_ds,
        tokenizer_src,
        tokenizer_tgt,
        max_len,
        print_msg,
        num_examples=2,
    ):
        self.model.eval()
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
                encoder_input = batch["encoder_input"].to(self.device)  # (b, seq_len)
                encoder_mask = batch["encoder_mask"].to(self.device)  # (b, 1, 1, seq_len)

                # check that the batch size is 1
                assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"

                model_out = self.greedy_decode(
                    source=encoder_input,
                    source_mask=encoder_mask,
                    tokenizer_src=tokenizer_src,
                    tokenizer_tgt=tokenizer_tgt,
                    max_len=max_len,
                )

                source_text = batch["src_text"][0]
                target_text = batch["tgt_text"][0]
                model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

                source_texts.append(source_text)
                expected.append(target_text)
                predicted.append(model_out_text)

                # Print the source, target and model output
                print_msg("-" * console_width)
                print_msg(f"{'SOURCE: ':>12}{source_text}")
                print_msg(f"{'TARGET: ':>12}{target_text}")
                print_msg(f"{'PREDICTED: ':>12}{model_out_text}")

                if count == num_examples:
                    print_msg("-" * console_width)
                    break
                pass
            pass

        if self.writer:
            # Evaluate the character error rate
            # Compute the char error rate
            metric = torchmetrics.CharErrorRate()
            cer = metric(predicted, expected)
            self.writer.add_scalar("validation cer", cer, self.global_step)
            self.writer.flush()

            # Compute the word error rate
            metric = torchmetrics.WordErrorRate()
            wer = metric(predicted, expected)
            self.writer.add_scalar("validation wer", wer, self.global_step)
            self.writer.flush()

            # Compute the BLEU metric
            metric = torchmetrics.BLEUScore()
            bleu = metric(predicted, expected)
            self.writer.add_scalar("validation BLEU", bleu, self.global_step)
            self.writer.flush()

    pass
