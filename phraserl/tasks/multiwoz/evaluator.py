from phraserl.utils.metrics import Metric
from .original.evaluator import OrigMultiWozEvaluator, BLEUScorer


class MultiWozEvaluator(Metric):
    mode = "max"

    def __init__(self):
        self.outputs = {}
        self.orig_evaluator = OrigMultiWozEvaluator()

    def __str__(self):
        if len(self.outputs) == 0:
            return "N/A"
        succ, match = self.orig_evaluator.evaluateModel(self.outputs)
        return f"[ Success Rate: {succ}, Inform Rate: {match} ]"

    def reset(self):
        self.outputs = {}

    def record(self, _input, output, eod):
        fn = _input["filenames"][0]
        sent = output["sent"]
        self.outputs.setdefault(fn, []).append(" ".join(sent))

    def value(self):
        succ, _ = self.orig_evaluator.evaluateModel(self.outputs)
        return float(succ)

    def reward(self, eod):
        if not eod:
            return 0.0
        succ, match = self.orig_evaluator.evaluateModel(self.outputs, soft_acc=True)
        self.reset()
        return float(succ) + float(match)


class MultiWozBleu(Metric):
    mode = "max"

    def __init__(self, soft_acc=False):
        self.orig_evaluator = BLEUScorer()
        self.refs = []
        self.hyps = []

    def __str__(self):
        if len(self.hyps) == 0:
            return "N/A"
        return "{:.3f}".format(self.orig_evaluator.score(self.hyps, self.refs))

    def reset(self):
        self.refs = []
        self.hyps = []

    def record(self, _input, output, eod):
        ref = _input["lbls"]  # (1, seqlen)
        hyp = output["sent"]  # (seqlen,)
        self.refs.append(ref)
        self.hyps.append(hyp)

    def value(self):
        if len(self.hyps) == 0:
            return 0
        return self.orig_evaluator.score(self.hyps, self.refs)

    def reward(self, eod):
        reward = self.value()
        self.reset()
        return reward
