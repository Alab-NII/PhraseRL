import os
from phraserl.utils.recorder import Recorder
from phraserl.utils.logging import logger
from .common import initialize, get_agent, get_task
from .test import test


def train(opt, agent, train_task, valid_task):
    n_epoch = opt.n_epoch
    impatience = 0
    agent_metrics = Recorder(agent.metrics, eval_metric_name=opt.eval_metric)
    task_metrics = Recorder(valid_task.get_metrics())

    if opt.get("notebook", False):
        from tqdm.notebook import tqdm
    else:
        from tqdm import tqdm

    for e in range(n_epoch):
        epoch = e + 1
        logger.info("epoch {} | training".format(epoch))

        """
        training
        """
        train_task.create_batches(opt.batch_size, shuffle=opt.shuffle_data)
        n_batches = train_task.n_batches()
        desc = "[Epoch {}]".format(epoch)
        with tqdm(train_task.batch_generator(), total=n_batches, desc=desc) as pbar:
            for batch, eod in pbar:
                output = agent.batch(batch, eod, train=True)
                agent_metrics.record(batch, output, eod)

        # show metrics
        logger.info("epoch {} | {}".format(epoch, agent_metrics))
        agent_metrics.reset()

        """
        validation
        """
        if (epoch) % int(opt.valid_freq) != 0:
            continue

        logger.info("epoch {} | validation".format(epoch))

        # agent metrics
        valid_task.create_batches(opt.batch_size)
        for batch, eod in valid_task.batch_generator():
            output = agent.batch(batch, eod, train=False)
            agent_metrics.record(batch, output, eod)
        logger.info("epoch {} | {}".format(epoch, agent_metrics))

        # task_metrics
        if len(task_metrics) > 0 and opt.get("run_task_metrics", False):
            valid_task.create_batches(1)
            for obs, eod in valid_task.batch_generator():
                output = agent.act(obs)
                task_metrics.record(obs, output, eod)
            logger.info("epoch {} | {}".format(epoch, task_metrics))

        # patience
        if agent_metrics.is_best():
            impatience = 0
            logger.info("epoch {} | Saving best model.".format(epoch))
            agent.save_model(os.path.join(opt.output_path, "best_model.pt"))
        elif opt.patience > 0:
            impatience += 1
            logger.info(
                "epoch {} | Did not beat the best model. "
                "impatience: {}".format(epoch, impatience)
            )
            if impatience >= opt.patience:
                logger.info("Ran out of patience.")
                break
        else:
            logger.info("epoch {} | Did not beat the best model.".format(epoch))

        agent_metrics.reset()

    agent.save_model(os.path.join(opt.output_path, f"final_model_epoch_{epoch}.pt"))
    logger.info("Finished training.")


def run(opt):
    # train
    device = initialize(opt, "train")
    train_task = get_task(opt, "train")
    valid_task = get_task(opt, "valid")
    agent = get_agent(opt, train_task.domain, train_task.get_vocab(), device)
    train(opt, agent, train_task, valid_task)

    # test
    opt.model_path = "best_model.pt"
    test_task = get_task(opt, "test")
    agent = get_agent(opt, train_task.domain, train_task.get_vocab(), device)
    test(opt, agent, test_task)
