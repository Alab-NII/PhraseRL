import pickle
import torch
import torch.nn as nn
import torch.optim as optim

from phraserl.utils.metrics import FloatMetric
from ..utils.reinforce import Reinforce
from ..utils.loader import load_batchifier
from ..base_agent import BaseAgent
from .model import TemplateModel

OPTIMS = {
    "adam": optim.Adam,
    "sgd": optim.SGD,
}
TEMPLATE_FN = "template.pkl"


class TemplateAgent(BaseAgent):
    def __init__(self, opt, domain, vocab, device=None):
        super().__init__(opt, domain, vocab, device)
        self.eos_idx = self.vocab.w2i(self.vocab.eos_token)
        self.bsz = self.opt.batch_size

        self.metrics = {
            "token_loss": FloatMetric(target="loss", mode="min"),
        }

        self.model = self.get_model()

        self.batchify = load_batchifier(opt)

        optim_cls = OPTIMS[self.opt.optim]
        self.optim = optim_cls(self.model.parameters(), lr=self.opt.lr)

        self.temps = []
        self.temps_info = []

        if self.opt.get("rl_optim") and self.opt.get("rl_lr", 0) > 0:
            optim_cls = OPTIMS[self.opt.rl_optim]
            self.rl_optim = optim_cls(self.model.parameters(), lr=self.opt.rl_lr)
        if self.opt.get("gamma", None) is not None:
            self.reinforce = Reinforce(self.device, self.opt.gamma)
        self.action_lps = []
        self.rewards = []

    def get_model(self):
        model = TemplateModel(self.opt, self.device, self.domain, self.vocab).to(
            self.device
        )
        return model

    """
    Corpus training.
    """

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

            self.model.detach_state()

        if (
            eod
            or self.opt.get("reset_every_turn", False)
            or self.opt.get("autoencoder", False)
        ):
            self.model.reset_state()

        output = {"loss": it_loss}
        return output

    def act(self, obs, train=False, template=None, act_emb=None, with_eos=False):
        batch = self.batchify(obs, self.domain, self.vocab, self.device)

        if train:
            self.model.train()
        else:
            self.model.eval()

        out, states, state_lens, state_lps, act_emb = self.model.generate(
            batch, template=template, act_emb=act_emb, sample_state=train
        )

        if train:
            self.action_lps.extend(state_lps)

        sent = []
        for token in out:
            if token == self.eos_idx:
                if with_eos:
                    sent.append(token)
                break
            sent.append(token)
        sent = self.vocab.i2w(sent)

        if self.opt.get("reset_every_turn", False) or self.opt.get(
            "autoencoder", False
        ):
            self.model.reset_state()

        output = {
            "sent": sent,
            "states": states,
            "state_lens": state_lens,
            "act_emb": act_emb,
        }
        return output

    def reset(self):
        self.model.reset_state()

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(
            torch.load(path, map_location=self.device), strict=False
        )

    """
    Template extractions.
    """

    def extract_templates(self, batch, eod):
        raw_batch = batch
        batch = self.batchify(batch, self.domain, self.vocab, self.device)
        self.model.eval()

        temps, temps_lens, act_emb = self.model.extract_templates(batch)

        batches = self._batch_split(raw_batch)
        temps_info = [
            {"temp": t, "lens": l, "act_emb": a, "batch": b}
            for t, l, a, b in zip(temps, temps_lens, act_emb, batches)
        ]
        self.temps.extend(temps)
        self.temps_info.extend(temps_info)
        if eod or self.opt.reset_every_turn:
            self.model.reset_state()

    def save_templates(self, path):
        with open(path, "wb") as f:
            pickle.dump({"temps": self.temps, "temps_info": self.temps_info}, f)

    def load_templates(self, path):
        with open(path, "rb") as f:
            t = pickle.load(f)
            self.temps = t["temps"]
            self.temps_info = t["temps_info"]

    def _batch_split(self, batch):
        # TODO: Only built for MultiWOZ
        bsz = len(batch["txts"])
        batches = []
        for i in range(bsz):
            txt = [batch["txts"][i]]
            lbl = [batch["lbls"][i]]
            ctx = {
                "bs": [batch["ctxs"]["bs"][i]],
                "db": [batch["ctxs"]["db"][i]],
            }
            fn = [batch["filenames"][i]]
            utt_idx = batch["utt_idx"]
            batches.append(
                {
                    "txts": txt,
                    "lbls": lbl,
                    "ctxs": ctx,
                    "filenames": fn,
                    "utt_idx": utt_idx,
                }
            )
        return batches

    """
    Policy training.
    """

    def policy_update(self, reward, eod):
        new_reward_len = len(self.action_lps) - len(self.rewards)
        new_reward = [0.0 for _ in range(new_reward_len)]
        new_reward[-1] = float(reward)
        self.rewards.extend(new_reward)

        if not eod:
            return

        loss = self.reinforce.forward(self.rewards, self.action_lps)

        torch.autograd.set_detect_anomaly(True)
        self.rl_optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.opt.clip)
        self.rl_optim.step()

        self.rewards = []
        self.action_lps = []
