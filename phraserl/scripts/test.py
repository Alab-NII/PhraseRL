from phraserl.utils.recorder import Recorder
from phraserl.utils.logging import logger
from .common import initialize, get_task, get_agent


def test(opt, agent, test_task):
    if opt.get("notebook", False):
        from tqdm.notebook import tqdm
    else:
        from tqdm import tqdm

    logger.info("testing")

    agent_metrics = Recorder(agent.metrics)
    task_metrics = Recorder(test_task.get_metrics())

    # agent metrics
    if not opt.get("task_only", False):
        test_task.create_batches(opt.batch_size)
        n_batches = test_task.n_batches()
        for batch, eod in tqdm(test_task.batch_generator(), total=n_batches):
            output = agent.batch(batch, eod, train=False)
            agent_metrics.record(batch, output, eod)
        logger.info("Agent metrics - {}".format(agent_metrics))

    # task metrics
    if len(task_metrics) > 0:
        test_task.create_batches(1)
        n_batches = test_task.n_batches()
        for obs, eod in tqdm(test_task.batch_generator(), total=n_batches):
            output = agent.act(obs)
            task_metrics.record(obs, output, eod)
            if eod:
                agent.reset()
        logger.info("Task metrics - {}".format(task_metrics))


def run(opt):
    device = initialize(opt, "test")
    train_task = get_task(opt, "train")
    test_task = get_task(opt, "test")
    agent = get_agent(opt, train_task.domain, train_task.get_vocab(), device)

    test(opt, agent, test_task)
