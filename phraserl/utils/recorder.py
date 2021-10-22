from phraserl.utils.metrics import Bleu
from phraserl.tasks.multiwoz.evaluator import MultiWozBleu, MultiWozEvaluator

TASK_METRICS = {
    "bleu": Bleu,
    "multiwoz": MultiWozEvaluator,
    "multiwoz-bleu": MultiWozBleu,
}


def get_task_metrics(name_list):
    metrics = {}
    for name in name_list:
        metrics[name] = TASK_METRICS[name]()
    return metrics


class Recorder:
    def __init__(self, metrics, eval_metric_name=None, coefs=None):
        """
        Args:
            metrics: dict with names as keys and Metric as values
        """
        self.metrics = metrics
        self.eval_metric_name = eval_metric_name
        self.best_value = None
        self.coefs = [1.0 for _ in range(len(self.metrics))] if coefs is None else coefs

    def __str__(self):
        metric_strs = []
        for name, metric in self.metrics.items():
            metric_strs.append("{}: {}".format(name, metric))
        return ", ".join(metric_strs)

    def __len__(self):
        return len(self.metrics)

    def record(self, input_, output, eod):
        for metric in self.metrics.values():
            metric.record(input_, output, eod)

    def reset(self):
        for metric in self.metrics.values():
            metric.reset()

    def is_best(self):
        if self.eval_metric_name not in self.metrics.keys():
            return False

        eval_metric = self.metrics[self.eval_metric_name]
        if (
            self.best_value is None
            or (eval_metric.mode == "min" and eval_metric.value() < self.best_value)
            or (eval_metric.mode == "max" and eval_metric.value() > self.best_value)
        ):
            self.best_value = eval_metric.value()
            return True

        return False

    def reward(self, eod):
        reward = 0
        for metric, coef in zip(self.metrics.values(), self.coefs):
            reward += metric.reward(eod) * coef
        return reward
