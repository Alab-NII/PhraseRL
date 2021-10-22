import torch
import statistics


class Reinforce:
    def __init__(self, device, gamma, max_all_returns=1000):
        self.device = device
        self.gamma = gamma
        self.max_all_returns = max_all_returns
        self.all_returns = []

    def forward(self, rewards, action_lps):
        ret = 0
        rets = []
        for r in rewards[::-1]:
            ret = r + self.gamma * ret
            rets.insert(0, ret)
        self.all_returns.extend(rets)
        self.all_returns = self.all_returns[-self.max_all_returns :]

        rets = torch.tensor(rets).to(self.device)
        rets = (rets - statistics.mean(self.all_returns)) / (
            statistics.stdev(self.all_returns) + 1e-10
        )

        lps = torch.stack(action_lps)
        loss = (-lps * rets).sum() / len(rets)

        return loss
