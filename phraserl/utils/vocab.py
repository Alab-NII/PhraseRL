DEFAULT_SP_TOKENS = [
    ("pad_token", "<pad>"),
    ("unk_token", "<unk>"),
    ("bos_token", "<s>"),
    ("eos_token", "</s>"),
]


class Vocab:
    def __init__(
        self, sp_tokens=DEFAULT_SP_TOKENS, cased=False,
    ):
        self.cased = cased
        self.word2id = {}
        self.sp_tokens = []

        for name, token in sp_tokens:
            self.add_sp_token(name, token)

        if not hasattr(self, "unk_token"):
            raise ValueError("'unk_token' must be set to special tokens")

        self.word2id.setdefault(self.unk_token, len(self.word2id))
        self.id2word = {v: k for k, v in self.word2id.items()}

    def __len__(self):
        return len(self.word2id)

    def add_sp_token(self, name, token):
        self.word2id[token] = len(self.word2id)
        self.sp_tokens.append(token)
        setattr(self, name, token)

    def build_vocab(self, sents, cutoff=-1):
        word_cnt = {}

        def count(sent):
            nonlocal word_cnt
            for word in sent:
                if isinstance(word, list):
                    # if word is a list go deeper
                    count(word)
                    continue
                if not self.cased and word not in self.sp_tokens:
                    word = word.lower()
                word_cnt[word] = word_cnt.get(word, 0) + 1

        count(sents)

        n_vocab = 0
        n_unk = 0
        for word, cnt in sorted(word_cnt.items(), key=lambda x: -x[1]):
            n_vocab += cnt
            if cutoff != -1 and cnt <= cutoff:
                n_unk += cnt
                continue
            if not self.cased and word not in self.sp_tokens:
                word = word.lower()
            _id = len(self.word2id)
            self.word2id.setdefault(word, _id)
            self.id2word[_id] = word

        return n_vocab, n_unk

    def w2i(self, word):
        if isinstance(word, list) or isinstance(word, tuple):
            return [self.w2i(w) for w in word]
        else:
            if not self.cased and word not in self.sp_tokens:
                word = word.lower()
            return self.word2id.get(word, self.word2id[self.unk_token])

    def i2w(self, _id):
        if isinstance(_id, list) or isinstance(_id, tuple):
            return [self.i2w(i) for i in _id]
        else:
            return self.id2word[_id]
