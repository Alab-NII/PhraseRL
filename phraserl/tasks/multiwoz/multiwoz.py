import os
import json
import random

from phraserl.utils.vocab import Vocab
from phraserl.utils.logging import logger

from ..base_task import BaseTask, BaseDomain
from .evaluator import MultiWozEvaluator, MultiWozBleu

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
EOS_TOKEN = "<eos>"
YOU_TOKEN = "YOU:"
THEM_TOKEN = "THEM:"

SP_TOKENS_TO_IDX = [
    ("pad_token", PAD_TOKEN),
    ("unk_token", UNK_TOKEN),
    ("eos_token", EOS_TOKEN),
    ("you_token", YOU_TOKEN),
    ("them_token", THEM_TOKEN),
]

DATA_FILE_NAME = {
    "train": "train_dials.json",
    "valid": "val_dials.json",
    "test": "test_dials.json",
}


class MultiWozDomain(BaseDomain):
    bs_size = 94
    db_size = 30


class MultiWozTask(BaseTask):
    domain = MultiWozDomain

    def __init__(self, opt, datatype="train"):
        super().__init__(opt, datatype)

        # Lazy loading
        self.batches = None
        self.eods = None
        self.vocab = None
        self.bleu = None
        self.evaluator = None

        logger.info("Building {} data.".format(datatype))

        self.data_path = opt["data_path"]

        # Loading data from file
        with open(os.path.join(self.data_path, DATA_FILE_NAME[datatype]), "r") as f:
            d = json.load(f)
            self.data = []
            for i, (k, v) in enumerate(d.items()):
                if os.environ.get("DEBUG", False) and i >= opt.batch_size:
                    # Only load one batch for debugging
                    break
                v["filename"] = k
                self.data.append(v)

    def get_metrics(self):
        self.bleu = self.bleu if self.bleu else MultiWozBleu()
        self.evaluator = self.evaluator if self.evaluator else MultiWozEvaluator()
        return {"bleu": self.bleu, "multiwoz": self.evaluator}

    def get_vocab(self):
        if self.vocab is not None:
            return self.vocab

        cutoff = self.opt.vocab_cutoff

        logger.info(
            "Building vocab in {} dataset with cutoff {}.".format(self.datatype, cutoff)
        )

        sents = []
        for d in self.data:
            for utt in d["sys"] + d["usr"]:
                sents.append(utt.strip().split())

        vocab = Vocab(SP_TOKENS_TO_IDX, cased=False)
        n_vocab, n_unk = vocab.build_vocab(sents, cutoff=cutoff)

        ratio = n_unk / (n_vocab + n_unk) * 100
        logger.info(
            "{} dataset vocab: # of vocabs {}, unk {}, unk ratio {:.2f}%.".format(
                self.datatype, n_vocab, n_unk, ratio
            )
        )

        self.vocab = vocab  # Cache
        return vocab

    def create_batches(self, bsz, shuffle=False):
        if shuffle:
            random.shuffle(self.data)

        self.batches = []
        self.eods = []

        n_data = len(self.data)

        eps_idxs = list(range(bsz))
        utt_idx = 0
        next_idx = bsz
        eods = [False for _ in range(bsz)]
        while next_idx <= n_data:
            if utt_idx == 0:
                eps_idxs.sort(key=lambda x: len(self.data[x]["usr"]), reverse=True)

            batch_filenames = []  # using this for evaluation
            batch_txts = []
            batch_lbls = []
            batch_ctxs = {"bs": [], "db": []}

            for j in range(bsz):
                eps = eps_idxs[j]
                d = self.data[eps]

                if eods[j]:
                    fn = ""
                    txt = []
                    lbl = []
                    bs = []
                    db = []
                else:
                    fn = d["filename"]
                    txt = d["usr"][utt_idx].strip().split()
                    lbl = d["sys"][utt_idx].strip().split()
                    bs = d["bs"][utt_idx]
                    db = d["db"][utt_idx]
                    eods[j] = True if utt_idx + 1 >= len(d["usr"]) else False

                batch_filenames.append(fn)
                batch_txts.append(txt)
                batch_lbls.append(lbl)
                batch_ctxs["bs"].append(bs)
                batch_ctxs["db"].append(db)

            batch = {
                "filenames": batch_filenames,
                "utt_idx": utt_idx,
                "txts": batch_txts,
                "lbls": batch_lbls,
                "ctxs": batch_ctxs,
            }
            self.batches.append(batch)

            if sum(eods) == bsz:
                utt_idx = 0
                eps_idxs = [next_idx + i for i in range(bsz)]
                next_idx += bsz
                eods = [False for _ in range(bsz)]
                is_eod = 1
            else:
                utt_idx += 1
                is_eod = 0

            self.eods.append(is_eod)

    def n_batches(self):
        if self.batches is None:
            raise RuntimeError("create_batches must be called before n_batches")
        return len(self.batches)

    def batch_generator(self):
        if self.batches is None:
            raise RuntimeError("create_batches must be called before batch_generator")

        for batch, eod in zip(self.batches, self.eods):
            yield batch, eod
