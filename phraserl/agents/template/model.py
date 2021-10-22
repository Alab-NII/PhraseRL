import heapq
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.distributions import Categorical

from ..utils.loader import load_context_encoder
from .act_encoder import CategoricalActEncoder, ContinuousActEncoder, ContextOnly


class TemplateModel(nn.Module):
    """
    Neural template generation model for dialog.
    (c.f. Wiseman et al. 2018, https://arxiv.org/abs/1808.10122)
    """

    def __init__(self, opt, device, domain, vocab):
        super().__init__()
        self.opt = opt
        self.device = device
        self.domain = domain
        self.vocab = vocab
        n_vocab = len(self.vocab)

        self.pad_idx = self.vocab.w2i(self.vocab.pad_token)
        self.eos_idx = self.vocab.w2i(self.vocab.eos_token)
        self.bos_idx = self.vocab.w2i(self.vocab.you_token)
        self.eop_idx = n_vocab  # end-of-phrase index

        # parameters
        word_emb_dim = self.opt.word_emb_dim
        state_emb_dim = self.opt.state_emb_dim
        rnn_hid_dim = self.opt.rnn_hid_dim
        dropout = self.opt.dropout
        trans_ab_dim = self.opt.trans_ab_dim
        cd_lin_dim = self.opt.cd_lin_dim
        self.trans_cd_dim = self.opt.trans_cd_dim
        self.k_mul = self.opt.k_mul
        self.k_base = self.opt.k_base
        self.k_states = self.k_base * self.k_mul
        self.l_max = self.opt.l_max

        # ctx
        self.ctx_encoder, ctx_emb_dim = load_context_encoder(opt, domain)
        opt["ctx_emb_dim"] = ctx_emb_dim

        # encoder
        self.state_embedding = nn.Embedding(self.k_base, state_emb_dim)
        self.word_embedding = nn.Embedding(
            n_vocab, word_emb_dim, padding_idx=self.pad_idx
        )
        self.encoder = nn.GRU(word_emb_dim, rnn_hid_dim)

        # latent speech act
        if opt.get("act_type", "") == "categorical":
            self.act_encoder = CategoricalActEncoder(self.device, opt)
        elif opt.get("act_type", "") == "context_only":
            self.act_encoder = ContextOnly(self.device, opt, ctx_emb_dim)
        else:
            self.act_encoder = ContinuousActEncoder(self.device, opt)
        self.act_emb_dim = self.act_encoder.output_dim

        # state initialization
        self.init_lin = nn.Sequential(
            nn.Linear(self.act_emb_dim, rnn_hid_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(rnn_hid_dim, self.k_states),
        )

        # transition
        self.trans_A = nn.Parameter(
            torch.empty(self.k_states, trans_ab_dim).to(self.device)
        )
        self.trans_B = nn.Parameter(
            torch.empty(trans_ab_dim, self.k_states).to(self.device)
        )
        self.trans_lin_CD = nn.Sequential(
            nn.Linear(self.act_emb_dim, cd_lin_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(cd_lin_dim, self.k_states * self.trans_cd_dim * 2),
        )
        nn.init.xavier_uniform_(self.trans_A)
        nn.init.xavier_uniform_(self.trans_B)
        if self.opt.get("disallow_self_trans", False):
            self.self_trans_filter = torch.diag(
                torch.full((self.k_states,), -float("inf"))
            ).to(self.device)

        # length
        if not self.opt.unif_len_ps:
            self.len_lin = nn.Linear(self.act_emb_dim, self.l_max)

        # emission
        self.rnn_init_lin = nn.Linear(self.act_emb_dim, rnn_hid_dim)
        self.seg_rnn = nn.GRU(state_emb_dim + word_emb_dim, rnn_hid_dim)
        self.seg_decoder = nn.Sequential(
            nn.Linear(rnn_hid_dim, rnn_hid_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(rnn_hid_dim, n_vocab + 1),  # add one for <eop>
        )
        if self.opt.autoregressive:
            self.ar_rnn = nn.GRU(word_emb_dim, rnn_hid_dim)

        # softmax
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.softmax = nn.Softmax(dim=-1)

        self.last_h = None
        self.n_step = 0

    def forward(self, batch):
        """
        Train neural template generation.
        """
        txt = batch["txts"]  # (seqlen, bsz)
        txt_lens = batch["txt_lens"]
        lbl = batch["lbls"]  # (seqlen, bsz)
        lbl_lens = batch["lbl_lens"]
        ctx = batch.get("ctxs", None)

        # get dialog act from contexts
        if self.opt.get("autoencoder", False):
            _, self.last_h = self.read(lbl, lbl_lens, self.last_h)
        else:
            _, self.last_h = self.read(txt, txt_lens, self.last_h)
        last_h = self.last_h.squeeze(0)
        ctx_emb = self.ctx_encoder(ctx)
        act_emb, act_loss = self.act_encoder(last_h, ctx_emb)  # (bsz, act_emb_dim)

        # get distributions
        init_lps = self._init_dist(act_emb)  # (bsz, k)
        trans_lps = self._transition_dist(act_emb)  # (bsz, k, k)
        len_lps = self._length_dist(act_emb)  # (l, bsz, k)
        emis_lps = self._emission_dist(lbl, act_emb)  # (l, seqlen, bsz, k)

        pad_mask = (lbl != self.pad_idx).to(torch.float).to(self.device)

        beta_star = self._hsmm_backward(trans_lps, len_lps, emis_lps, pad_mask)
        lse = (beta_star[0] + init_lps).logsumexp(1)
        recon_loss = -lse.sum() / lse.size(0)

        loss = recon_loss + act_loss

        # read label
        _, self.last_h = self.read(lbl, lbl_lens, self.last_h, add_bos=True)

        self.n_step += 1

        return loss

    def generate(self, batch, template=None, act_emb=None, sample_state=False):
        txt = batch["txts"]
        txt_lens = batch["txt_lens"]
        lbl = batch.get("lbls", None)
        lbl_lens = batch.get("lbl_lens", None)
        ctx = batch.get("ctxs", None)
        bsz = txt.size(1)

        assert bsz == 1

        out = []
        states = []
        state_lens = []
        state_lps = []

        # get dialog act from contexts
        if self.opt.get("autoencoder", False):
            _, self.last_h = self.read(lbl, lbl_lens, self.last_h)
        else:
            _, self.last_h = self.read(txt, txt_lens, self.last_h)
        last_h = self.last_h.squeeze(0)
        ctx_emb = self.ctx_encoder(ctx)

        if act_emb is None:
            act_emb, _ = self.act_encoder(last_h, ctx_emb)  # (bsz, act_emb_dim)

        init_lps = self._init_dist(act_emb)  # (bsz, k)
        trans_lps = self._transition_dist(act_emb)  # (bsz, k, k)

        # initial state
        if template is not None:
            template_idx = 0
            cur_state = torch.tensor([template[template_idx]], dtype=torch.long)
            cur_state = cur_state.to(self.device)
            is_final_state = template_idx == len(template) - 1
        else:
            if sample_state:
                m = Categorical(init_lps.exp())
                cur_state = m.sample()  # (bsz,)
            else:
                _, cur_state = init_lps.topk(1)
                cur_state = cur_state.squeeze(1)
            states.append(cur_state.item())
            state_lps.append(init_lps[0, cur_state.item()])

        # prepare initial x and h
        init_x = torch.full((bsz,), self.bos_idx, dtype=torch.long).to(self.device)
        init_h = self.rnn_init_lin(act_emb).unsqueeze(0)  # (1, bsz, rnn_hid_dim)
        init_ar_h = None

        eos = False
        while not eos:
            state_emb = self.state_embedding((cur_state % self.k_base).detach())
            # (bsz, state_emb_dim)

            pq = [
                (0, [init_x.detach()], init_h.detach(), init_ar_h)
            ]  # (log prob, outputs, hidden, ar hidden)
            tmp_pq = []
            while len(pq) > 0:
                fin = 0
                for _ in range(self.opt.beam_size):
                    if len(pq) == 0:
                        break

                    prev_nlp, prev_xs, h, ar_h = heapq.heappop(pq)
                    x = prev_xs[-1]

                    if (
                        x.item() in [self.eop_idx, self.eos_idx]
                        or len(prev_xs) > self.l_max
                    ):
                        fin += 1
                        heapq.heappush(tmp_pq, (prev_nlp, prev_xs, h, ar_h))
                        continue

                    word_emb = self.word_embedding(x)  # (bsz, word_emb_dim)
                    x = torch.cat([word_emb, state_emb], dim=1).unsqueeze(0)
                    x, h = self.seg_rnn(x, h)  # (1, bsz, rnn_emb_dim)
                    if self.opt.autoregressive:
                        ar_x, ar_h = self.ar_rnn(word_emb.unsqueeze(0), ar_h)
                        x = x + ar_x
                    token_logits = self.seg_decoder(x.squeeze(0))
                    token_lps = self.log_softmax(token_logits)  # (bsz, n_vocab)

                    if len(prev_xs) == self.l_max + 1:  # force <eop>
                        if template is not None and is_final_state:
                            token_lps[:, self.eop_idx] = -float("inf")
                        next_x = torch.tensor([self.eop_idx], dtype=torch.long).to(
                            self.device
                        )
                        next_lp = token_lps[:, self.eop_idx].item()
                        n = len(prev_xs) - 1
                        new_nlp = (prev_nlp * n - next_lp) / (n + 1)
                        new_xs = prev_xs + [next_x]
                        heapq.heappush(tmp_pq, (new_nlp, new_xs, h, ar_h))
                        continue

                    next_lps, next_xs = token_lps.topk(self.opt.beam_size)
                    # (bsz, beam_size)
                    for next_lp, next_x in zip(next_lps[0], next_xs[0]):
                        next_lp = next_lp.item()
                        if template is not None:
                            if (is_final_state and next_x == self.eop_idx) or (
                                not is_final_state and next_x == self.eos_idx
                            ):
                                next_lp = -float("inf")
                        n = len(prev_xs) - 1
                        new_nlp = (prev_nlp * n - next_lp) / (n + 1)
                        new_xs = prev_xs + [next_x.unsqueeze(0).detach()]
                        heapq.heappush(tmp_pq, (new_nlp, new_xs, h, ar_h))

                pq = tmp_pq
                if fin >= self.opt.beam_size:
                    break

            init_ar_h = pq[0][3]

            best_seg = pq[0][1][1:]  # starts from <bos>
            state_len = 0
            for x in best_seg:
                if x.item() == self.bos_idx:
                    continue
                if x.item() == self.eop_idx:
                    break
                out.append(x.item())
                state_len += 1
                if x.item() == self.eos_idx or len(out) >= self.opt.max_gen_len:
                    eos = True
                    break
            state_lens.append(state_len)

            # state transition
            if template is not None:
                template_idx += 1
                if len(template) <= template_idx:
                    break
                cur_state = torch.tensor([template[template_idx]], dtype=torch.long)
                cur_state = cur_state.to(self.device)
                is_final_state = template_idx == len(template) - 1
            else:
                prev_state = cur_state.item()
                if sample_state:
                    m = Categorical(trans_lps[:, prev_state].exp())
                    cur_state = m.sample()  # (bsz,)
                else:
                    _, cur_state = trans_lps[:, prev_state].topk(1)
                    cur_state = cur_state.squeeze(1)
                state_lps.append(trans_lps[0, prev_state, cur_state.item()])

            states.append(cur_state.item())

        if lbl is not None:
            # read label
            _, self.last_h = self.read(lbl, lbl_lens, self.last_h, add_bos=True)
        else:
            # read self output
            _out = torch.tensor(out, dtype=torch.long).to(self.device).unsqueeze(1)
            out_lens = torch.tensor([len(out)], dtype=torch.long).to(self.device)
            _, self.last_h = self.read(_out, out_lens, self.last_h, add_bos=True)

        return out, states, state_lens, state_lps, act_emb.squeeze(0)

    def extract_templates(self, batch):
        """
        Extract templates from batches.

        Returns:
            templates: sequences of (start_idx, end_idx, state)
        """
        txt = batch["txts"]  # (seqlen, bsz)
        txt_lens = batch["txt_lens"]
        lbl = batch["lbls"]  # (seqlen, bsz)
        lbl_lens = batch["lbl_lens"]
        ctx = batch.get("ctxs", None)

        # get dialog act from contexts
        if self.opt.get("autoencoder", False):
            _, self.last_h = self.read(lbl, lbl_lens, self.last_h)
        else:
            _, self.last_h = self.read(txt, txt_lens, self.last_h)
        last_h = self.last_h.squeeze(0)
        ctx_emb = self.ctx_encoder(ctx)
        act_emb, _ = self.act_encoder(last_h, ctx_emb)

        # get distributions
        init_lps = self._init_dist(act_emb)  # (bsz, k)
        trans_lps = self._transition_dist(act_emb)  # (bsz, k, k)
        len_lps = self._length_dist(act_emb)  # (l, bsz, k)
        emis_lps = self._emission_dist(lbl, act_emb)  # (l, seqlen, bsz, k)

        # viterbi
        temps, temp_lens = self._viterbi(init_lps, trans_lps, len_lps, emis_lps, lbl)

        # read label
        _, self.last_h = self.read(lbl, lbl_lens, self.last_h, add_bos=True)

        return temps, temp_lens, act_emb

    def get_act_emb(self, batch):
        txt = batch["txts"]
        txt_lens = batch["txt_lens"]
        lbl = batch.get("lbls", None)
        lbl_lens = batch.get("lbl_lens", None)
        ctx = batch.get("ctxs", None)

        if self.opt.get("autoencoder", False):
            _, h = self.read(lbl, lbl_lens, self.last_h)
        else:
            _, h = self.read(txt, txt_lens, self.last_h)
        ctx_emb = self.ctx_encoder(ctx)

        act_emb, _ = self.act_encoder(h.squeeze(0), ctx_emb)  # (bsz, act_emb_dim)
        return act_emb

    def read(self, x, x_lens, h=None, add_bos=False):
        """
        Read text.

        Args:
            x: Input text.
            x_lens: Lengths of input text.
            h: Hidden state.
        """
        if add_bos:
            bsz = x.size(1)
            bos = torch.full((1, bsz), self.bos_idx, dtype=torch.long)
            bos = bos.to(self.device)
            x = torch.cat([bos, x], dim=0)
            x_lens = x_lens + 1
        x_emb = self.word_embedding(x)
        packed_x_emb = pack_padded_sequence(x_emb, x_lens, enforce_sorted=False)
        x, h = self.encoder(packed_x_emb, h)
        x, _ = pad_packed_sequence(x)
        return x, h

    def _init_dist(self, act_emb):
        """
        Args:
            enc_h: (bsz, rnn_hid_dim)
            ctx_emb: (bsz, ctx_emb_dim)

        Returns:
            init_lps: (bsz, k_states)
        """
        init_logits = self.init_lin(act_emb)
        init_lps = self.log_softmax(init_logits)
        return init_lps

    def _transition_dist(self, act_emb):
        """
        Args:
            act_emb: (bsz, act_emb_dim)

        Returns:
            trans_lps: (bsz, k_states, k_states)
        """
        bsz = act_emb.size(0)

        # C x D
        cd = self.trans_lin_CD(act_emb)
        cd = cd.view(bsz, self.k_states, -1)
        trans_C, trans_D = cd[:, :, : self.trans_cd_dim], cd[:, :, self.trans_cd_dim :]
        trans_score = torch.bmm(trans_C, trans_D.transpose(1, 2))

        # A x B
        if self.opt.get("use_ab_score", True):
            ab_score = torch.mm(self.trans_A, self.trans_B)
            ab_score = ab_score.unsqueeze(0).expand(bsz, self.k_states, self.k_states)
            trans_score = ab_score + trans_score

        if self.opt.get("disallow_self_trans", False):
            trans_score = trans_score + self.self_trans_filter
        trans_lps = self.log_softmax(trans_score)

        return trans_lps

    def _length_dist(self, act_emb):
        """
        Args:
            act_emb: (bsz, act_emb_size)

        Returns:
            len_lps: (l, bsz, k_states)
        """
        bsz = act_emb.size(0)

        if self.opt.unif_len_ps:
            len_score = torch.ones(bsz, self.k_states, self.l_max).to(self.device)
        else:
            states = torch.tensor(range(self.k_states), dtype=torch.long)
            states = states.to(self.device).unsqueeze(0)
            states = self.state_embedding(states.expand(bsz, self.k_states))
            act = act_emb.unsqueeze(1).expand(bsz, self.k_states, -1)
            len_score = self.len_lin(torch.cat([states, act], dim=2))

        len_lps = self.log_softmax(len_score).permute(2, 0, 1)
        return len_lps

    def _emission_dist(self, lbl, act_emb):
        """
        Args:
            lbl: (seqlen, bsz)
            act_emb: (bsz, act_emb_dim)

        Returns:
            emis_lps: (l_max, seqlen, bsz, k_states)
        """
        seqlen, bsz = lbl.size()

        pad = torch.tensor([[[self.pad_idx]]], dtype=torch.long).to(self.device)
        seg = self._to_seg(lbl.unsqueeze(2), pad)[:-1]
        bos = torch.full((1, seg.size(1)), self.bos_idx, dtype=torch.long)
        bos = bos.to(self.device)
        seg = torch.cat([bos, seg.squeeze(2)], dim=0)
        seg_emb = self.word_embedding(seg)  # (l_max + 1, bsz * seqlen, word_emb_dim)

        # initial hidden
        init_hid = self.rnn_init_lin(act_emb)
        init_hid = init_hid.unsqueeze(0).unsqueeze(2)  # (1, bsz, 1, rnn_hid_dim)
        init_hid = init_hid.expand(-1, bsz, seqlen, -1).contiguous()
        init_hid = init_hid.view(init_hid.size(0), bsz * seqlen, -1)

        if self.opt.autoregressive:
            bos = torch.full((1, bsz), self.bos_idx, dtype=torch.long)
            bos = bos.to(self.device)
            lbl_emb = self.word_embedding(torch.cat([bos, lbl[:-1]], dim=0))
            ar_x, _ = self.ar_rnn(lbl_emb)  # (seqlen + 1, bsz, rnn_hid_dim)

            pad = torch.zeros((1, 1, ar_x.size(2)), dtype=torch.float)
            pad = pad.to(self.device)
            ar_x = self._to_seg(ar_x, pad)  # (l_max + 1, bsz * seqlen, rnn_hid_dim)

        emis_lps = []
        for k in range(self.k_base):
            state = torch.tensor([[k]], dtype=torch.long).to(self.device)
            state_emb = self.state_embedding(state)
            state_emb = state_emb.expand(seg_emb.size(0), seg_emb.size(1), -1)

            x, _ = self.seg_rnn(torch.cat([seg_emb, state_emb], dim=2), init_hid)
            if self.opt.autoregressive:
                x = x + ar_x

            seg_logits = self.seg_decoder(x)
            seg_lps = self.log_softmax(seg_logits)
            # seg_lps: (l_max + 1, bsz * seqlen, n_words)

            # take the predicted probabilities of the label words
            pred_lps = seg_lps[:-1].gather(2, seg[1:].unsqueeze(2)).squeeze(2)
            pred_cumsum_lps = pred_lps.cumsum(dim=0)  # (l_max, bsz * seqlen)

            # add log probability of the end-of-phrase token
            eop_lps = seg_lps[1:, :, self.eop_idx]  # (l_max, bsz * seqlen)
            pred_cumsum_lps = pred_cumsum_lps + eop_lps

            emis_lps.append(pred_cumsum_lps)

        emis_lps = torch.stack(emis_lps).view(self.k_base, self.l_max, bsz, seqlen)
        emis_lps = emis_lps.permute(1, 3, 2, 0)
        if self.k_mul > 1:
            emis_lps = emis_lps.repeat(1, 1, 1, self.k_mul)

        return emis_lps

    def _to_seg(self, x, pad):
        """
        Convert to segments.
        [ 1  5 ] becomes [ 1   2   3   4   5   6   7  <p>]
        [ 2  6 ]         [ 2   3   4  <p>  6   7  <p> <p>]
        [ 3  7 ]         [ 3   4  <p> <p>  7  <p> <p> <p>]
        [ 4 <p>]         [ 4  <p> <p> <p> <p> <p> <p> <p>]
        if l_max = 3.

        Args:
            x: (seqlen, bsz, n)
            pad: (1, 1, n)

        Returns:
            seg: (l_max + 1, bsz * seqlen, n)
        """
        seqlen, bsz, _ = x.size()

        pad = pad.expand(self.l_max, bsz, -1).to(self.device)
        padded_x = torch.cat([x, pad], dim=0)

        seg = [padded_x[i : i + self.l_max + 1] for i in range(seqlen)]
        seg = torch.stack(seg).permute(1, 2, 0, 3).contiguous()
        seg = seg.view(self.l_max + 1, bsz * seqlen, -1)

        return seg

    def _hsmm_backward(self, trans_lps, len_lps, emis_lps, pad_mask):
        """
        A dynamic programming backward method for hidden semi-markov model.

        Args:
            trans_lps: (bsz, k_states, k_states)
            len_lps: (l_max, bsz, k_states)
            emis_lps: (l_max, seqlen, bsz, k_states)
            pad_mask: (seqlen, bsz)

        Returns:
            beta_star: (seqlen + 1, bsz, k_states)
        """
        _, seqlen, bsz, _ = emis_lps.size()
        pad_mask = pad_mask.unsqueeze(2).expand(-1, -1, self.k_states)

        beta = [None] * (seqlen + 1)
        beta_star = [None] * (seqlen + 1)
        beta[seqlen] = torch.zeros(bsz, self.k_states).to(self.device)

        for t in range(seqlen - 1, -1, -1):
            steps_fwd = min(seqlen - t, self.l_max)

            beta_star_terms = (
                torch.stack(beta[t + 1 : t + 1 + steps_fwd])
                + emis_lps[:steps_fwd, t]
                + len_lps[:steps_fwd]
            )  # (steps_fwd, bsz, k_states)
            beta_star[t] = beta_star_terms.logsumexp(0) * pad_mask[t]

            beta_terms = (
                beta_star[t].unsqueeze(1).expand(-1, self.k_states, -1) + trans_lps
            )  # (bsz, k_states, k_states)
            beta[t] = beta_terms.logsumexp(2) * pad_mask[t]

        return beta_star

    def _viterbi(self, init_lps, trans_lps, len_lps, emis_lps, lbl):
        """
        Viterbi algorithm for getting the most probable sequence.

        Args:
            init_lps: (bsz, k_states)
            trans_lps: (bsz, k_states, k_states)
            len_lps: (l_max, bsz, k_states)
            emis_lps: (l_max, seqlen, bsz, k_states)
            lbl: (seqlen, bsz)

        Returns:
            seqs: list of sequences
        """
        _, seqlen, bsz, _ = emis_lps.size()
        neginf = -float("inf")

        bwd_emis_lps = self._to_bwd_emis_lps(emis_lps)
        flipped_len_lps = torch.flip(len_lps, (0,))

        delta = torch.full((seqlen + 1, bsz, self.k_states), neginf).to(self.device)
        delta_star = torch.full((seqlen + 1, bsz, self.k_states), neginf)
        delta_star = delta_star.to(self.device)
        delta_star[0] = init_lps

        bps = torch.full((seqlen + 1, bsz, self.k_states), self.l_max, dtype=torch.long)
        bps = bps.to(self.device)
        bps_star = torch.zeros(seqlen + 1, bsz, self.k_states, dtype=torch.long)
        bps_star = bps_star.to(self.device)
        bps_star[0] = (
            torch.arange(0, self.k_states)
            .unsqueeze(0)
            .expand(bsz, self.k_states)
            .to(self.device)
        )

        for t in range(1, seqlen + 1):
            steps_back = min(self.l_max, t)

            delta_terms = (
                delta_star[t - steps_back : t]
                + bwd_emis_lps[-steps_back:, t - 1]
                + flipped_len_lps[-steps_back:]
            )

            maxes, argmaxes = torch.max(delta_terms, 0)
            delta[t] = maxes.squeeze(0)
            bps[t].fill_(self.l_max)  # magic
            bps[t] = bps[t] - argmaxes.squeeze(0)
            if steps_back < self.l_max:
                bps[t] = bps[t] - (self.l_max - steps_back)

            delta_t = delta[t].unsqueeze(2).expand(bsz, self.k_states, self.k_states)
            delta_t = delta_t.transpose(0, 1)
            delta_star_terms = trans_lps.transpose(0, 1) + delta_t

            maxes, argmaxes = torch.max(delta_star_terms, 0)
            delta_star[t] = maxes.squeeze(0)
            bps_star[t] = argmaxes.squeeze(0)

        # Recovering backpointers
        seqs = []
        seq_lens = []
        for b in range(bsz):
            seq = []
            seq_len = []
            _, state = delta[seqlen, b].max(0)
            state = state.item()
            cur_idx = seqlen
            while cur_idx > 0:
                last_len = bps[cur_idx, b, state].item()
                if lbl[cur_idx - last_len, b].item() != self.pad_idx:
                    seq.append(state)
                    seq_len.append(last_len)
                cur_idx -= last_len
                state = bps_star[cur_idx, b, state].item()
            seqs.append(seq[::-1])
            seq_lens.append(seq_len[::-1])

        return seqs, seq_lens

    def _to_bwd_emis_lps(self, emis_lps):
        """
        Convert to backward emission logprobs.
        from [  p0   p1   p2   p3   p4  ] to [ -inf -inf p0:2 p1:3 p2:4 ]
             [ p0:1 p1:2 p2:3 p3:4 p4:5 ]    [ -inf p0:1 p1:2 p2:3 p3:4 ]
             [ p0:2 p1:3 p2:4 p3:5 p4:6 ]    [  p0   p1   p2   p3   p4  ]

        Args:
            emis_lps: (l_max, seqlen, bsz, k_states)

        Returns:
            bwd_emis_lps: (l_max, seqlen, bsz, k_states)
        """
        neginf = -float("inf")
        bwd_emis_lps = torch.full_like(emis_lps, neginf).to(self.device)
        bwd_emis_lps[self.l_max - 1] = emis_lps[0]
        for _l in range(1, self.l_max):
            bwd_emis_lps[self.l_max - _l - 1, _l:] = emis_lps[_l, :-_l]
        return bwd_emis_lps

    def reset_state(self):
        self.last_h = None

    def detach_state(self):
        if self.last_h is not None:
            self.last_h = self.last_h.detach()
