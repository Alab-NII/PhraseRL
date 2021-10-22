import os
from phraserl.scripts.common import initialize, get_agent, get_task

TEMPLATE_FN = "template.pkl"


def extract_templates(opt, agent, train_task):
    if opt.get("notebook", False):
        from tqdm.notebook import tqdm
    else:
        from tqdm import tqdm
    train_task.create_batches(opt.batch_size, shuffle=opt.shuffle_data)
    n_batches = train_task.n_batches()
    with tqdm(train_task.batch_generator(), total=n_batches) as pbar:
        for batch, eod in pbar:
            agent.extract_templates(batch, eod)
    agent.save_templates(os.path.join(opt.output_path, TEMPLATE_FN))


def run(opt):
    # train
    device = initialize(opt, opt.data_type)
    train_task = get_task(opt, "train")
    target_task = (
        train_task if opt.data_type == "train" else get_task(opt, opt.data_type)
    )
    agent = get_agent(opt, train_task.domain, train_task.get_vocab(), device)
    extract_templates(opt, agent, target_task)
