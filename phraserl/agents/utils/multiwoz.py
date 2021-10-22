import torch


def pad_sequence(seqs, pad_idx=0):
    maxlen = max([len(seq) for seq in seqs])
    for i in range(len(seqs)):
        seqs[i] += [pad_idx] * (maxlen - len(seqs[i]))
    return seqs


def multiwoz_batchify(raw_batch, domain, vocab, device):
    them_idx = vocab.w2i(vocab.them_token)
    pad_idx = vocab.w2i(vocab.pad_token)
    eos_idx = vocab.w2i(vocab.eos_token)

    # txts
    raw_txts = vocab.w2i(raw_batch["txts"])
    txts = []
    for raw_txt in raw_txts:
        if len(raw_txt) == 0:
            # If the dialog had ended, the length of txt is 0
            break
        # Insert THEM: at the beginning
        raw_txt.insert(0, them_idx)
        txts.append(raw_txt)
    txt_lens = torch.tensor([len(txt) for txt in txts], dtype=torch.long).to(device)
    # txts is a tensor (seqlen, bsz)
    txts = (
        torch.tensor(pad_sequence(txts, pad_idx), dtype=torch.long)
        .to(device)
        .transpose(0, 1)
    )

    # lbls
    lbls = None
    lbl_lens = None
    if raw_batch.get("lbls", None) is not None:
        raw_lbls = vocab.w2i(raw_batch["lbls"])
        lbls = []
        for raw_lbl in raw_lbls:
            if len(raw_lbl) == 0:
                # If the dialog had ended, the length of lbl is 0
                break
            # Append <eos>
            raw_lbl.append(eos_idx)
            lbls.append(raw_lbl)
        lbl_lens = torch.tensor([len(lbl) for lbl in lbls], dtype=torch.long).to(device)
        # lbls is a tensor (seqlen, bsz)
        lbls = (
            torch.tensor(pad_sequence(lbls, pad_idx), dtype=torch.long)
            .to(device)
            .transpose(0, 1)
        )

    bs = [x for x in raw_batch["ctxs"]["bs"] if len(x) > 0]
    bs = torch.tensor(bs, dtype=torch.float).to(device)
    db = [x for x in raw_batch["ctxs"]["db"] if len(x) > 0]
    db = torch.tensor(db, dtype=torch.float).to(device)

    batch = {
        "utt_idx": raw_batch["utt_idx"],
        "filenames": raw_batch["filenames"],
        "txts": txts,  # (seqlen, bsz)
        "txt_lens": txt_lens,  # (bsz,)
        "lbls": lbls,  # (seqlen, bsz)
        "lbl_lens": lbl_lens,  # (bsz,)
        "ctxs": {"bs": bs, "db": db},
    }

    return batch


class MultiWozContextEncoder:
    def __call__(self, ctx):
        ctx = torch.cat([ctx["bs"], ctx["db"]], dim=1)
        return ctx
