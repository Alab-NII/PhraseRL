import os
from tqdm import tqdm
from collections import OrderedDict

from phraserl.utils.recorder import Recorder, get_task_metrics
from .common import initialize, get_agent, get_task

N_AVE = 5000


def policy(opt, agent, train_task):
    reward_metrics = Recorder(
        get_task_metrics(opt.rewards), coefs=opt.get("coefs", None)
    )

    eps_rewards = []
    best_ave_reward = 0.0
    for epoch in range(opt.n_epoch):
        train_task.create_batches(1, shuffle=True)
        n_batches = train_task.n_batches()
        desc = f"[Epoch {epoch}]"

        with tqdm(train_task.batch_generator(), total=n_batches, desc=desc) as pbar:
            eps_reward = 0.0
            for obs, eod in pbar:
                output = agent.act(obs, train=True)
                reward_metrics.record(obs, output, eod)
                reward = reward_metrics.reward(eod)

                agent.policy_update(reward, eod)

                eps_reward += reward
                if eod:
                    agent.reset()

                    eps_rewards.append(eps_reward)
                    ave_reward = sum(eps_rewards[-N_AVE:]) / len(eps_rewards[-N_AVE:])
                    if len(eps_rewards) > N_AVE and best_ave_reward < ave_reward:
                        best_ave_reward = ave_reward
                        agent.save_model(os.path.join(opt.output_path, "best_model.pt"))

                    pbar.set_postfix(
                        OrderedDict(episode=len(eps_rewards), ave_reward=ave_reward)
                    )
                    eps_reward = 0.0

                    if len(eps_rewards) % 5000 == 0:
                        model_path = os.path.join(
                            opt.output_path, f"policy_model_eps_{len(eps_rewards)}.pt"
                        )
                        agent.save_model(model_path)

        with open(os.path.join(opt.output_path, "rewards.csv"), "w") as f:
            f.writelines([str(r) + "\n" for r in eps_rewards])


def run(opt):
    device = initialize(opt, "policy")
    train_task = get_task(opt, "train")
    agent = get_agent(opt, train_task.domain, train_task.get_vocab(), device)

    policy(opt, agent, train_task)
