from transformers import T5Tokenizer, T5ForConditionalGeneration
from pytorch_lightning import LightningModule
import torch
import math
from torch import nn
import metrics
from transformers.optimization import Adafactor


class Seq2SeqT5(LightningModule):
    def __init__(self, model="google-t5/t5-small", max_len=15, lr=1e-3, tokenizer=None, device=None):
        super().__init__()
        self.max_len = max_len
        self.t5_model = T5ForConditionalGeneration.from_pretrained(model)
        if tokenizer is None:
            self.tokenizer = T5Tokenizer.from_pretrained("google-t5/t5-small")
        else:
            self.tokenizer=tokenizer
            self.t5_model.resize_token_embeddings(len(tokenizer))
        self.lr = lr
        
    def training_step(self, batch):
        input_ids, attention_mask, target_ids = batch
        loss = self.t5_model(input_ids=input_ids, attention_mask=attention_mask, labels=target_ids).loss

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch):
        input_ids, attention_mask, target_ids = batch
        out = self.t5_model(input_ids=input_ids, attention_mask=attention_mask, labels=target_ids)

        predicted_ids = torch.argmax(out.logits, dim=-1)
        bleu_score, _, _ = metrics.bleu_scorer(target_ids, predicted_ids, self.tokenizer)
        self.log("bleu_score", bleu_score, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        self.log("val_loss", out.loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return out.loss

    def predict(self, input_ids, attention_mask):
        output = self.t5_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_len,
            num_beams=5,
        )
        return output

    def configure_optimizers(self):
        a_opt = Adafactor(self.parameters(), lr=self.lr, relative_step=False)
        return a_opt

    def eval_bleu(self, predicted_ids_list, target_tensor):
        predicted = torch.stack(predicted_ids_list)
        predicted = predicted.squeeze().detach().cpu().numpy().swapaxes(0, 1)[:, 1:]
        actuals = target_tensor.squeeze().detach().cpu().numpy()[:, 1:]
        bleu_score, actual_sentences, predicted_sentences = metrics.bleu_scorer(
            predicted=predicted, actual=actuals, target_tokenizer=self.target_tokenizer
        )
        return bleu_score, actual_sentences, predicted_sentences




