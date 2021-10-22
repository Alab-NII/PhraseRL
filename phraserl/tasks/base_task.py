class BaseDomain:
    pass


class BaseTask:
    def __init__(self, opt, datatype="train"):
        self.opt = opt
        self.datatype = datatype
        self.metrics = {}

    def get_metrics(self):
        pass

    def get_vocab(self):
        pass

    def n_batches(self):
        pass

    def create_batches(self, bsz, shuffle=False):
        pass

    def batch_generator(self):
        pass


class BaseScenarioGenerator:
    def __init__(self, opt):
        self.opt = opt

    def __iter__(self):
        return self

    def __next__(self):
        raise NotImplementedError


class Scenario:
    def __init__(self):
        pass
