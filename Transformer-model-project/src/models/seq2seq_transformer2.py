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

    def forward(self, src): 
        _, N = src.shape
        emb_src = self.pos_enc(self.enc_emb(src) * math.sqrt(self.emb_dim))
        
        tgt = torch.full((1, N), self.tgt_sos_idx).to(self.device)
        tgt_tokens = [torch.full((N,), self.tgt_sos_idx).long()]
        tgt_dist = [nn.functional.one_hot(tgt_tokens[0], self.linear.out_features).to(self.device).float()]
        
        tgt_dist[0] = tgt_dist[0].masked_fill(tgt_dist[0] == 0, float('-inf')) \
                                 .masked_fill(tgt_dist[0] == 1, float(0))
        
        for _ in range(self.max_len - 1):
            tgt_seq_len = tgt.size(0)
            
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_seq_len).to(self.device)  
            emb_tgt = self.pos_enc(self.enc_emb(tgt) * math.sqrt(self.emb_dim))
            
            out = self.linear(self.transformer(emb_src, emb_tgt, tgt_mask=tgt_mask)[-1])
            
            next_token = torch.argmax(out, dim=-1)
            
            tgt = torch.cat((tgt, next_token.unsqueeze(0)), dim=0).long()
            tgt_dist.append(out)
            
        return tgt, tgt_dist

    def training_step(self, batch):
        src, tgt = batch # (N, S), (N, T)
        src = src.transpose(0, 1) # (S, N)
        out, out_dist = self.forward(src)
        out = out.transpose(0, 1)
        
        out_dist = torch.stack(out_dist, dim=0).permute(1, 0, 2).reshape(-1, self.tgt_dim)
        tgt = tgt.reshape(-1)
        
        loss = self.criterion(out_dist, tgt)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch):
        src, tgt = batch # (N, S), (N, T)
        src = src.transpose(0, 1) # (S, N)
        out, out_dist = self.forward(src) # (N, T)
        out = out.transpose(0, 1)
        
        bleu_score, _, _ = metrics.bleu_scorer(tgt, out, self.tgt_tokenizer)
        self.log("bleu_score", bleu_score, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        out_dist = torch.stack(out_dist, dim=0).permute(1, 0, 2).reshape(-1, self.tgt_dim)
        tgt = tgt.reshape(-1)
        
        loss = self.criterion(out_dist, tgt)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def predict(self, phrase):
        self.eval()
        prediction, dist = self.forward(torch.tensor([self.src_tokenizer(phrase)]).transpose(0, 1).to(self.device))
        self.train()
        return " ".join(self.tgt_tokenizer.decode(prediction.reshape(-1)))
    
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
