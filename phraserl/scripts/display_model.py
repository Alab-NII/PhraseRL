from phraserl.utils.logging import logger

from .common import initialize, get_task, get_agent


def display_model(opt, agent, test_task):
    test_task.create_batches(1, shuffle=True)
    batch_gen = test_task.batch_generator()
    for _ in range(opt.n_display):
        obs, eod = batch_gen.__next__()
        txt = obs["txts"][0]
        lbl = obs["lbls"][0]

        output = agent.act(obs)
        sent = output["sent"]

        txt = " ".join(txt)
        lbl = " ".join(lbl)
        model = " ".join(sent)

        logger.info("----------------------------------------------------------------")
        logger.info(f"Text  : {txt}")
        logger.info(f"Label : {lbl}")
        logger.info(f"Model : {model}")
        if "states" in output:
            logger.info(f"States: {output['states']}")

        if eod:
            agent.reset()

    logger.info("----------------------------------------------------------------")


def run(opt):
    opt.batch_size = 1

    device = initialize(opt, "display-model")
    train_task = get_task(opt, "train")
    test_task = get_task(opt, "test")
    agent = get_agent(opt, train_task.domain, train_task.get_vocab(), device)

    display_model(opt, agent, test_task)
