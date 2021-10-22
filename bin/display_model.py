import argparse

from common import load_opt
import phraserl.scripts.display_model as display_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", "-c", nargs="*", type=str)
    parser.add_argument("--output", "-o", required=True, type=str)
    parser.add_argument("--model", "-m", default="best_model.pt", type=str)
    parser.add_argument("--seed", "-s", default=0, type=int)
    parser.add_argument("--n_display", "-n", default=20, type=int)
    args = parser.parse_args()

    opt = load_opt(args.conf, args.output)

    opt["model_path"] = args.model
    opt["random_seed"] = args.seed
    opt["n_display"] = args.n_display

    display_model.run(opt)
