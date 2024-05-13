from pytorch_lightning import LightningModule
import torch
import math
from torch import nn
import metrics

# https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class PositionalEncoding(LightningModule):
    def __init__(
        self, 
        d_model: int,
        dropout: float = 0.1,
        max_len: int = 5000
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class Seq2SeqTransformer(LightningModule):
    def __init__(self, 
        lr=1e-3,
        nhead=8,
        src_dim=512, tgt_dim=512,
        emb_dim=512, hdn_dim=512,
        enc_nlayers=3, dec_nlayers=3,
        tgt_tokenizer=None,
        src_tokenizer=None,
        dropout=0.1, max_len=15,
        src_pad_idx=None, src_sos_idx=None, src_eos_idx=None,
        tgt_pad_idx=None, tgt_sos_idx=None, tgt_eos_idx=None,
    ):
        
        super().__init__()
        self.lr = lr
        
        self.tgt_tokenizer = tgt_tokenizer
        self.src_tokenizer = src_tokenizer
        
        self.tgt_dim = tgt_dim
        
        self.emb_dim = emb_dim
        self.max_len = max_len
        
        self.src_sos_idx = src_sos_idx
        self.src_eos_idx = src_eos_idx
        self.src_pad_idx = src_pad_idx
        
        self.tgt_sos_idx = tgt_sos_idx
        self.tgt_eos_idx = tgt_eos_idx
        self.tgt_pad_idx = tgt_pad_idx

        self.src_mask, self.tgt_mask = None, None
        
        self.enc_emb = nn.Embedding(num_embeddings=src_dim, embedding_dim=emb_dim)
        self.dec_emb = nn.Embedding(num_embeddings=tgt_dim, embedding_dim=emb_dim)

        self.pos_enc = PositionalEncoding(emb_dim, dropout, max_len)
        
        self.transformer = nn.Transformer(
            d_model=emb_dim,
            nhead=nhead,
            num_encoder_layers=enc_nlayers,
            num_decoder_layers=dec_nlayers,
            dim_feedforward=hdn_dim,
            dropout=dropout
        )

        self.linear = nn.Linear(in_features=emb_dim, out_features=tgt_dim)
        
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.tgt_pad_idx)

    def forward(self, src, tgt): 
        src_seq_len, N = src.shape
        tgt_seq_len, N = tgt.shape
        
        emb_src = self.pos_enc(self.enc_emb(src) * math.sqrt(self.emb_dim))
        emb_tgt = self.pos_enc(self.dec_emb(tgt) * math.sqrt(self.emb_dim))

        self.src_mask = nn.Transformer.generate_square_subsequent_mask(src_seq_len).to(self.device)
        self.tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_seq_len).to(self.device)
        
        out = self.transformer(emb_src, emb_tgt, src_mask=self.src_mask, tgt_mask=self.tgt_mask)
        
        out = self.linear(out)
        return out

    def training_step(self, batch):
        src, tgt = batch # (N, S), (N, T)
        src = src.transpose(0, 1) # (S, N)
        tgt = tgt.transpose(0, 1) # (T, N)
        
        out = self.forward(src, tgt[:-1]).permute(1, 0, 2) # (N, T-1, TV)
        tgt = tgt.transpose(0, 1) # (N, T)
        
        tgt_vocab_size = out.shape[-1]
        
        out = out.reshape(-1, tgt_vocab_size)
        tgt = tgt[:, 1:].reshape(-1)
        
        loss = self.criterion(out, tgt)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch):
        src, tgt = batch # (N, S), (N, T)
        src = src.transpose(0, 1) # (S, N)
        tgt = tgt.transpose(0, 1) # (T, N)
        
        out = self.forward(src, tgt[:-1]).permute(1, 0, 2) # (N, T-1, TV)
        tgt = tgt.transpose(0, 1) # (N, T)
        
        predicted_ids = torch.argmax(out, dim=-1)
        
        bleu_score, _, _ = metrics.bleu_scorer(tgt, predicted_ids, self.tgt_tokenizer)
        self.log("bleu_score", bleu_score, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        tgt_vocab_size = out.shape[-1]
        
        out = out.reshape(-1, tgt_vocab_size)
        tgt = tgt[:, 1:].reshape(-1)
        
        loss = self.criterion(out, tgt)
        
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def predict(self, src):
        self.eval()
        
        with torch.no_grad():
            tgt = torch.Tensor([[self.tgt_sos_idx]]).to(self.device)
            src = src.transpose(0,1)
    
            for i in range(self.max_len - 1):
                out = self.forward(src.long(), tgt.transpose(0,1).long())
                next_word = torch.argmax(out, dim=-1).reshape(-1).unsqueeze(0)[:, -1]
                tgt = torch.cat((tgt, next_word.unsqueeze(0)), dim=-1).long()
    
                if tgt[:, -1].item() == self.tgt_eos_idx:
                    break
    
        self.train()
        return tgt
    
    def configure_optimizers(self):
        a_opt = torch.optim.AdamW(self.parameters(), lr=self.lr)
        s_opt = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=a_opt,
            max_lr=1e-2,
            anneal_strategy='linear',
            epochs = 20,
            steps_per_epoch=1800
        )
        return [a_opt], [s_opt]
