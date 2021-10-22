from .multiwoz import MultiWozContextEncoder, multiwoz_batchify


def load_context_encoder(opt, domain):
    if opt.task == "multiwoz":
        ctx_emb_dim = domain.bs_size + domain.db_size
        return MultiWozContextEncoder(), ctx_emb_dim
    else:
        raise ValueError("not supported")


def load_batchifier(opt):
    if opt.task == "multiwoz":
        return multiwoz_batchify
    else:
        raise ValueError("not supported")
