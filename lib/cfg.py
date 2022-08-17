import os
import yaml


def load_config(configpath, skip_dir_creation=False):
    with open(configpath) as f:
        cfg = yaml.safe_load(f)

    experiment_id = os.path.splitext(os.path.basename(configpath))[0]
    cfg['experiment_id'] = experiment_id
    model_dir = os.path.join(cfg['output']['model_dir'], experiment_id)
    if not os.path.exists(model_dir) and not skip_dir_creation:
        os.makedirs(model_dir)
    return cfg, model_dir

