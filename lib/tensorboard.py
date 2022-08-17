import os
from torch.utils.tensorboard import SummaryWriter


def get_summary_writer(cfg, fold=None):
    if not cfg['output']['use_tensorboard']:
        return None

    experiment_id = cfg['experiment_id']
    if fold is not None:
        experiment_id += f'_f{fold}'
    log_dir = os.path.join(cfg['output']['log_dir'], experiment_id)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    return SummaryWriter(log_dir=log_dir)