import os
import toml
from attrdict import AttrDict

CONFIG_FILE_NAME = "config.toml"


def load_opt(conf, output):
    new_conf_path = os.path.join(output, CONFIG_FILE_NAME)

    if os.path.exists(new_conf_path):
        if conf:
            raise FileExistsError(
                f"{output} already exists. Run without designating a config file."
            )
        with open(new_conf_path, "r") as f:
            opt = AttrDict(toml.load(f))
        opt["output_path"] = output
        return opt

    if not conf:
        raise ValueError("You must provide at least one config file.")

    opt = AttrDict()
    for c in conf:
        with open(c, "r") as f:
            opt.update(toml.load(f))

    os.makedirs(output)
    with open(new_conf_path, "w") as f:
        toml.dump(opt, f)

    opt["output_path"] = output
    return opt
