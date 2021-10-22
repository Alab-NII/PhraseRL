class BaseAgent:
    def __init__(self, opt, domain, vocab, device=None):
        self.opt = opt
        self.domain = domain
        self.vocab = vocab
        self.device = device
        self.metrics = {}

    def batch(self, batch, eod, train=True):
        """
        Train the model on a batch.
        """
        raise NotImplementedError

    def act(self, obs, train=False):
        """
        Take an action.
        """
        raise NotImplementedError

    def reset(self):
        pass

    def save_model(self, path):
        raise NotImplementedError

    def load_model(self, path):
        raise NotImplementedError
