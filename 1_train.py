import os
from torch.utils.data import DataLoader

from lib.cfg import load_config
from lib.dataset import T1T2Dataset
from lib.models import load_seg_model
from lib.optimizers import load_optimizer
from lib.transforms import get_segmentation_transforms
from lib.training import load_criterion, save_model, cycle_pose
from lib.vis import vis_pose

import wandb

CONFIG = "experiments/001.yaml"

if __name__ == "__main__":

    # Load config
    cfg, model_dir = load_config(CONFIG)

    # Data
    train_transforms, test_transforms = get_segmentation_transforms(cfg)
    ds_train = T1T2Dataset(cfg, 'train', train_transforms)
    ds_test = T1T2Dataset(cfg, 'test', test_transforms)
    dl_train = DataLoader(ds_train, cfg['training']['batch_size'], shuffle=True,
                          num_workers=cfg['training']['num_workers'], pin_memory=True)
    dl_test = DataLoader(ds_test, cfg['training']['batch_size'], shuffle=False,
                         num_workers=1, pin_memory=True)

    # Model
    model, starting_epoch, state = load_seg_model(cfg)
    optimizer, scheduler = load_optimizer(model, cfg, state, steps_per_epoch=len(dl_train))
    train_criterion, test_criterion = load_criterion(cfg)

    # WandB
    wandb.init(project="t1t2", config=cfg, notes=cfg.get('notes', None))
    wandb.save("*.png")  # Write PNG files immediately to WandB
    wandb.watch(model)

    # Train
    best_loss, best_path, last_save_path = 1e10, None, None
    n_epochs = cfg['training']['n_epochs']

    for epoch in range(starting_epoch, n_epochs + 1):
        print(f"\nEpoch {epoch} of {n_epochs}")

        train_loss = cycle_pose('train', model, dl_train, epoch, train_criterion, optimizer, cfg, scheduler)
        test_loss = cycle_pose('test', model, dl_test, epoch, test_criterion, optimizer, cfg)

        # save model if required('all', 'best', or 'improvement')
        state = {'epoch': epoch + 1,
                 'state_dict': model.module.state_dict() if cfg['training']['dataparallel'] else model.state_dict(),
                 'optimizer': optimizer.state_dict(),
                 'scheduler': scheduler}
        save_path = os.path.join(model_dir, f"{epoch}_{test_loss:.07f}.pt")
        best_loss, last_save_path = save_model(state, save_path, test_loss, best_loss, cfg, last_save_path)

        # vis
        vis_pose(dl_test, model, epoch, cfg)

    save_path = os.path.join(model_dir, f"final_{n_epochs}_{test_loss:.07f}.pt")
    best_loss, last_save_path = save_model(state, save_path, test_loss, best_loss, cfg, last_save_path, final=True)
