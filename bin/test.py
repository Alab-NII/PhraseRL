import argparse

from common import load_opt
import phraserl.scripts.test as test

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", "-c", default="", type=str)
    parser.add_argument("--output", "-o", required=True, type=str)
    parser.add_argument("--model", "-m", default="best_model.pt", type=str)
    parser.add_argument("--seed", "-s", default=0, type=int)
    parser.add_argument("--task_only", "-t", action="store_true")
    args = parser.parse_args()

    opt = load_opt(args.conf, args.output)

    opt["model_path"] = args.model
    opt["random_seed"] = args.seed
    opt["task_only"] = args.task_only

    test.run(opt)
