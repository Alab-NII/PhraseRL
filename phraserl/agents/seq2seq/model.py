import random
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.distributions import Categorical

from ..utils.loader import load_context_encoder


class Seq2SeqModel(nn.Module):
    def __init__(self, opt, device, domain, n_vocab, bos_idx, eos_idx, pad_idx):
        super().__init__()
        self.opt = opt
        self.device = device
        self.domain = domain

        self.bos_idx = bos_idx
        self.eos_idx = eos_idx
        self.pad_idx = pad_idx

        self.rnn_hid_dim = self.opt.rnn_hid_dim
        word_emb_dim = self.opt.word_emb_dim
        num_layers = self.opt.num_layers
        dropout = self.opt.dropout
        output_dim = self.opt.rnn_hid_dim

        if self.opt.use_attention:
            # attention
            self.kv_split = nn.Linear(self.rnn_hid_dim, 2 * self.rnn_hid_dim)
            output_dim += self.rnn_hid_dim

        self.embedding = nn.Embedding(n_vocab, word_emb_dim, padding_idx=self.pad_idx)
        self.encoder = nn.GRU(word_emb_dim, self.rnn_hid_dim, num_layers)

        self.ctx_encoder, ctx_emb_dim = load_context_encoder(opt, domain)

        self.ctx_to_act = nn.Sequential(
            nn.Linear(self.rnn_hid_dim + ctx_emb_dim, self.rnn_hid_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(self.rnn_hid_dim, self.rnn_hid_dim),
        )

        self.decoder = nn.GRU(word_emb_dim, self.rnn_hid_dim, num_layers)
        self.output = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(output_dim, n_vocab),
        )

        self.cross_entropy = nn.CrossEntropyLoss(
            ignore_index=self.pad_idx, reduction="mean"
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, batch):
        txt = batch["txts"]
        txt_lens = batch["txt_lens"]
        lbl = batch["lbls"]
        ctx = batch["ctxs"]

        # read
        x = self.embedding(txt)
        x = pack_padded_sequence(x, txt_lens, enforce_sorted=False)
        x, h = self.encoder(x, None)
        enc_hs, _ = pad_packed_sequence(x)

        # context
        ctx_emb = self.ctx_encoder(ctx).unsqueeze(0)
        ctx_emb = ctx_emb.expand(h.size(0), -1, -1)
        h = self.ctx_to_act(torch.cat([h, ctx_emb], dim=2))

        # write
        bsz = lbl.size(1)
        bos = torch.full((1, bsz), self.bos_idx, dtype=torch.long).to(self.device)
        x = torch.cat([bos, lbl[:-1]], dim=0)

        loss = 0
        _x = x[0].unsqueeze(0)
        for i in range(x.size(0)):
            if random.random() < self.opt.teacher_forcing:
                _x = x[i].unsqueeze(0)

            _x = self.embedding(_x)
            _x, h = self.decoder(_x, h)

            if self.opt.use_attention:
                attn = self.attention(_x, enc_hs)
                _x = torch.cat([_x.squeeze(0), attn], dim=1)

            _x = self.output(_x)
            loss += self.cross_entropy(_x, lbl[i])
            p = self.softmax(_x)

            if self.opt.sample_output:
                m = Categorical(p)
                _x = m.sample().unsqueeze(0)  # (1, bsz)
            else:
                _, _x = p.topk(1)
                _x = _x.transpose(0, 1)  # (1, bsz)

        return loss / self.opt.batch_size

    def generate(self, batch):
        txt = batch["txts"]
        txt_lens = batch["txt_lens"]
        ctx = batch["ctxs"]
        bsz = txt.size(1)

        # read
        x = self.embedding(txt)
        x = pack_padded_sequence(x, txt_lens, enforce_sorted=False)
        x, h = self.encoder(x, None)
        enc_hs, _ = pad_packed_sequence(x)

        # context
        ctx_emb = self.ctx_encoder(ctx).unsqueeze(0)
        ctx_emb = ctx_emb.expand(h.size(0), -1, -1)
        h = self.ctx_to_act(torch.cat([h, ctx_emb], dim=2))

        # write
        x = torch.full((1, bsz), self.bos_idx, dtype=torch.long).to(self.device)
        outs = []
        for _ in range(self.opt.max_gen_len):
            x = self.embedding(x)
            x, h = self.decoder(x, h)

            if self.opt.use_attention:
                attn = self.attention(x, enc_hs)
                x = torch.cat([x.squeeze(0), attn], dim=1)

            x = self.output(x)
            p = self.softmax(x)

            if self.opt.sample_output:
                m = Categorical(p)
                x = m.sample().unsqueeze(0)
            else:
                _, x = p.topk(1)

            outs.append(x.squeeze(0))
            if x == self.eos_idx:
                break

        return outs

    def attention(self, q, enc_hs):
        """
        Args:
            q: (1, bsz, rnn_hid_dim)
            enc_hs: (seqlen, bsz, rnn_hid_dim)

        Returns:
            attn: (bsz, rnn_hid_dim)
        """

        q_t = q.transpose(0, 1)
        enc_hs_t = enc_hs.transpose(0, 1)

        kv = self.kv_split(enc_hs_t)  # (bsz, seqlen, rnn_hid_dim * 2)
        k = kv[:, :, : self.rnn_hid_dim]
        v = kv[:, :, self.rnn_hid_dim :]

        attn_score = torch.bmm(q_t, k.transpose(1, 2))
        attn_weight = self.softmax(attn_score)  # (bsz, 1, seqlen)
        attn = torch.bmm(attn_weight, v)  # (bsz, 1, rnn_hid_dim)

        return attn.squeeze(1)
