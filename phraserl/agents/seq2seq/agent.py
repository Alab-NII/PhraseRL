import torch
import torch.nn as nn
import torch.optim as optim

from phraserl.utils.metrics import FloatMetric
from ..utils.loader import load_batchifier
from ..base_agent import BaseAgent
from .model import Seq2SeqModel

OPTIMS = {
    "adam": optim.Adam,
    "sgd": optim.SGD,
}


class Seq2SeqAgent(BaseAgent):
    def __init__(self, opt, domain, vocab, device=None):
        super().__init__(opt, domain, vocab, device)

        self.bsz = self.opt.batch_size

        self.metrics = {
            "token_loss": FloatMetric(target="loss", mode="min"),
        }

        self.pad_idx = self.vocab.w2i(self.vocab.pad_token)
        self.them_idx = self.vocab.w2i(self.vocab.them_token)
        self.eos_idx = self.vocab.w2i(self.vocab.eos_token)
        self.you_idx = self.vocab.w2i(self.vocab.you_token)

        n_vocab = len(self.vocab)
        self.model = Seq2SeqModel(
            self.opt,
            self.device,
            self.domain,
            n_vocab,
            self.you_idx,
            self.eos_idx,
            self.pad_idx,
        ).to(self.device)

        self.batchify = load_batchifier(opt)

        optim_cls = OPTIMS[self.opt.optim]
        self.optim = optim_cls(self.model.parameters(), lr=self.opt.lr)

        self.templates = []  # sequence of hidden states

    def batch(self, batch, eod, train=True):
        batch = self.batchify(batch, self.domain, self.vocab, self.device)

        if train:
            self.model.train()
        else:
            self.model.eval()

        loss = self.model(batch)
        it_loss = loss.item()

        if train:
            self.optim.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.opt.clip)
            self.optim.step()

        if eod:
            self.reset()

        output = {"loss": it_loss}
        return output

    def act(self, obs, train=False):
        batch = self.batchify(obs, self.domain, self.vocab, self.device)

        self.model.eval()
        out = self.model.generate(batch)

        out = [token.item() for token in out]
        sent = []
        for token in out:
            if token == self.eos_idx:
                break
            sent.append(token)
        sent = self.vocab.i2w(sent)

        output = {"sent": sent}
        return output

    def reset(self):
        pass

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
