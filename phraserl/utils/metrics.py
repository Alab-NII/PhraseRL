import torch
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


class Metric:
    mode = None

    def __init__(self):
        pass

    def __str__(self):
        pass

    def reset(self):
        pass

    def record(self, input_, output, eod):
        pass

    def value(self):
        pass

    def reward(self, eod):
        return self.value()


class FloatMetric(Metric):
    def __init__(self, target, mode="max"):
        self.values = []
        self.target = target
        self.mode = mode

    def __str__(self):
        if len(self.values) == 0:
            return "N/A"
        avg = sum(self.values) / len(self.values)
        return "{:.3f}".format(avg)

    def reset(self):
        self.values = []

    def record(self, input_, output, eod):
        if output.get(self.target, None) is None:
            return
        self.values.append(output[self.target])

    def value(self):
        if len(self.values) == 0:
            return None
        avg = sum(self.values) / len(self.values)
        return avg


class Accuracy(Metric):
    mode = "max"

    def __init__(self, target, label):
        self.cnt = 0
        self.correct = 0
        self.target = target
        self.label = label

    def __str__(self):
        if self.cnt == 0:
            return "N/A"
        return "{:.3f}".format(self.correct / self.total * 100)

    def reset(self):
        self.cnt = 0
        self.correct = 0

    def record(self, input_, output, eod):
        if (
            output.get(self.target, None) is None
            or input_.get(self.label, None) is None
        ):
            return

        tgt = output[self.target]
        lbl = input_[self.label]

        def _record(tgt, lbl):
            tgt = tgt.tolist() if isinstance(tgt, torch.Tensor) else tgt
            lbl = lbl.tolist() if isinstance(lbl, torch.Tensor) else lbl
            if isinstance(tgt, list):
                assert isinstance(lbl, list), "target and label have different size"
                for t, l in zip(tgt, lbl):
                    _record(t, l)
            else:
                if tgt == lbl:
                    self.correct += 1
                self.cnt += 1

        _record(tgt, lbl)

    def value(self):
        if self.cnt == 0:
            return 0
        return self.correct / self.cnt


class Bleu(Metric):
    mode = "max"

    def __init__(self):
        self.total = 0
        self.cnt = 0
        self.smth = SmoothingFunction()

    def __str__(self):
        if self.cnt == 0:
            return "N/A"
        return "{:.3f}".format(self.total / self.cnt)

    def reset(self):
        self.total = 0
        self.cnt = 0

    def record(self, input_, output, eod):
        ref = input_["lbls"][0]  # (seqlen,)
        hyp = output["sent"]  # (seqlen,)
        for r, h in zip(ref, hyp):
            self.total += sentence_bleu([r], [h], smoothing_function=self.smth.method1)
            self.cnt += 1

    def value(self):
        if self.cnt == 0:
            return 0
        return self.total / self.cnt


class Diversity(Metric):
    mode = "max"

    # TODO
    def __init__(self, target):
        self.target = target

    def __str__(self):
        pass

    def reset(self):
        pass

    def record(self, output, eod):
        pass

    def value(self):
        pass
