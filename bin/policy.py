import argparse

from common import load_opt
import phraserl.scripts.policy as policy

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", "-c", nargs="*", type=str)
    parser.add_argument("--output", "-o", required=True, type=str)
    parser.add_argument("--model", "-m", default="", type=str)
    parser.add_argument("--seed", "-s", default=0, type=int)
    args = parser.parse_args()

    opt = load_opt(args.conf, args.output)

    opt["model_path"] = args.model
    opt["random_seed"] = args.seed

    policy.run(opt)
