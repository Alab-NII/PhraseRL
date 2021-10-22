import torch
import torch.nn as nn


class STGumbelSoftmax(nn.Module):
    EPS = 1e-20

    def __init__(self, device, tau=1.0):
        super().__init__()
        self.device = device
        self.tau = tau
        self.softmax = nn.Softmax(dim=-1)

    def sample_gumbel(self, shape):
        noise = torch.rand(shape).to(self.device)
        return -torch.log(-torch.log(noise + self.EPS) + self.EPS)

    def forward(self, logits):
        y = logits + self.sample_gumbel(logits.size())
        y = self.softmax(y / self.tau)

        _, idx = y.max(dim=-1)
        y_hard = torch.zeros_like(y).view(-1, y.size(-1))
        y_hard = y_hard.scatter(1, idx.view(-1, 1), 1)
        y_hard = y_hard.view(y.size())

        return (y_hard - y).detach() + y


class BatchPriorRegularization(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device

    def forward(self, q, p):
        """
        Args:
            q: (bsz, n_act, m_act)
            p: (n_act, m_act)
        """
        q_prime = q.transpose(0, 1).mean(dim=1)  # (n_act, m_act)
        log_q_prime = (q_prime + 1e-20).log()  # (n_act, m_act)
        log_p_prime = (p + 1e-20).log()  # (n_act, m_act)

        kl_bpr = (log_q_prime - log_p_prime) * q_prime
        kl_bpr = kl_bpr.sum()
        return kl_bpr


class BaseActEncoder(nn.Module):
    def __init__(self, device, opt):
        super().__init__()
        self.device = device
        self.opt = opt
        self.output_dim = 0

    def forward(self, enc_h, ctx_emb):
        raise NotImplementedError


class CategoricalActEncoder(BaseActEncoder):
    def __init__(self, device, opt):
        super().__init__(device, opt)
        self.n_act = opt.n_act
        self.m_act = opt.m_act

        input_dim = opt.rnn_hid_dim
        if opt.get("use_context", False):
            input_dim += opt.ctx_emb_dim

        self.model = nn.Sequential(
            nn.Linear(input_dim, opt.rnn_hid_dim),
            nn.ReLU(),
            nn.Dropout(p=opt.dropout),
            nn.Linear(opt.rnn_hid_dim, self.n_act * self.m_act),
        )
        self.softmax = nn.Softmax(dim=-1)
        self.bpr = BatchPriorRegularization(device)
        self.gumbel_softmax = STGumbelSoftmax(device)
        self.output_dim = self.n_act * self.m_act

    def forward(self, enc_h, ctx_emb):
        bsz = enc_h.size(0)

        x = enc_h
        if self.opt.use_context:
            x = torch.cat([x, ctx_emb], dim=1)
        y = self.model(x).view(bsz * self.n_act, self.m_act)

        kl_loss = 0
        if self.opt.get("use_bpr", False):
            q = self.softmax(y).view(bsz, self.n_act, self.m_act)
            # prior is a uniform distribution
            p = torch.full((self.n_act, self.m_act), 1 / self.m_act).to(self.device)
            kl_loss = self.bpr(q, p)

        y = self.gumbel_softmax(y).view(bsz, self.n_act * self.m_act)

        if self.opt.get("act_freeze", False):
            y = y.detach()

        return y, kl_loss


class ContinuousActEncoder(BaseActEncoder):
    def __init__(self, device, opt):
        super().__init__(device, opt)

        self.model = nn.Sequential(
            nn.Linear(opt.rnn_hid_dim + opt.ctx_emb_dim, opt.rnn_hid_dim),
            nn.ReLU(),
            nn.Dropout(p=opt.dropout),
        )
        self.output_dim = opt.rnn_hid_dim

    def forward(self, enc_h, ctx_emb):
        """
        Args:
            enc_h: (bsz, rnn_hid_dim)
            ctx_emb: (bsz, ctx_emb_dim)
        """
        return self.model(torch.cat([enc_h, ctx_emb], dim=1)), 0


class ContextOnly(BaseActEncoder):
    def __init__(self, device, opt, ctx_emb_dim):
        super().__init__(device, opt)
        self.output_dim = ctx_emb_dim

    def forward(self, enc_h, ctx_emb):
        return ctx_emb, 0
