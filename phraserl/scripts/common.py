import os
import random
import toml
import torch

from phraserl.utils.logging import logger
from phraserl.tasks import get_task_cls
from phraserl.agents import get_agent_cls


def initialize(opt, script_name):
    # set random seed
    if not hasattr(opt, "random_seed"):
        opt.random_seed = random.randint(1, 10000)
    random.seed(opt.random_seed)
    torch.manual_seed(opt.random_seed)
    torch.cuda.manual_seed_all(opt.random_seed)

    # set logger
    log_path = f"{opt.output_path}/{script_name}_{{time:YYYY-MM-DD_HH-mm-ss}}.log"
    logger.add(log_path)

    # dump options
    opt_str = toml.dumps(opt)[:-1]  # Remove last \n
    logger.info("Configs:\n" + opt_str)

    # set device
    if torch.cuda.is_available():
        logger.info("CUDA is available, using GPU.")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    return device


def get_task(opt, datatype):
    task_cls = get_task_cls(opt.task)
    return task_cls(opt, datatype=datatype)


def get_agent(opt, domain, vocab, device):
    agent_cls = get_agent_cls(opt.agent)
    agent = agent_cls(opt, domain, vocab, device)

    if not opt.get("model_path"):
        return agent

    model_path = ""
    if os.path.exists(opt.model_path):
        model_path = opt.model_path
    elif os.path.exists(os.path.join(opt.output_path, opt.model_path)):
        model_path = os.path.join(opt.output_path, opt.model_path)
    else:
        raise FileNotFoundError(opt.model_path)

    logger.info(f"Loading model from {model_path}.")
    opt.model_path = model_path
    agent.load_model(model_path)

    return agent
