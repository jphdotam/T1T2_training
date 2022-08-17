import os
import numpy as np
from collections import deque, OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

import wandb


def load_criterion(cfg):

    def get_criterion(name, class_weights):
        if name == 'crossentropy':
            print(f"CrossEntropy with class_weights: {class_weights}")
            return nn.CrossEntropyLoss(weight=class_weights)
        elif name == 'mse':
            return nn.MSELoss()
        else:
            raise ValueError()

    class_weights_train = cfg['training'].get('class_weights_train', None)
    class_weights_test = cfg['training'].get('class_weights_test', None)

    if class_weights_train:
        print(f"Using class weights {class_weights_train} for training")
        class_weights_train = torch.tensor(class_weights_train).float().to(cfg['training']['device'])
    else:
        class_weights_train = None

    if class_weights_test:
        print(f"Using class weights {class_weights_test} for testing")
        class_weights_test = torch.tensor(class_weights_test).float().to(cfg['training']['device'])
    else:
        class_weights_test = None

    train_criterion = get_criterion(cfg['training']['train_criterion'], class_weights_train)
    test_criterion = get_criterion(cfg['training']['test_criterion'], class_weights_test)

    return train_criterion, test_criterion


class Am:
    "Simple average meter which stores progress as a running average"

    def __init__(self, n_for_running_average=100):  # n is in samples not batches
        self.n_for_running_average = n_for_running_average
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.running = deque(maxlen=self.n_for_running_average)
        self.running_average = -1

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.running.extend([val] * n)
        self.count += n
        self.avg = self.sum / self.count
        self.running_average = sum(self.running) / len(self.running)


def upsample_pred_if_needed(batch_y_pred, batch_y_true):
    ph, pw = batch_y_pred.size(-2), batch_y_pred.size(-1)
    h, w = batch_y_true.size(-2), batch_y_true.size(-1)
    if ph != h or pw != w:
        batch_y_pred = F.upsample(
            input=batch_y_pred, size=(h, w), mode='bilinear')
        return batch_y_pred
    else:
        return batch_y_pred


def cycle_pose(train_or_test, model, dataloader, epoch, criterion, optimizer, cfg, scheduler=None):
    log_freq = cfg['output']['log_freq']
    device = cfg['training']['device']
    mixed_precision = cfg['training'].get('mixed_precision', False)
    aux_loss_weight = cfg['training'].get('aux_loss', False)
    meter_loss = Am()

    if train_or_test == 'train':
        model.train()
        training = True
    elif train_or_test == 'test':
        model.eval()
        training = False
    else:
        raise ValueError(f"train_or_test must be 'train', or 'test', not {train_or_test}")

    def get_loss(y_p, y_t, crit, aux_loss_wt):
        if aux_loss_weight:
            loss_aux = crit(y_p['aux'], y_t)
            loss_out = crit(y_p['out'], y_t)
            return (loss_aux * aux_loss_wt) + loss_out
        else:
            if type(y_p) == OrderedDict:
                y_p = y_p['out']
            return crit(y_p, y_t)

    for i_batch, (x, y_true, _filepath) in enumerate(dataloader):
        # Forward pass
        optimizer.zero_grad()

        x = x.to(device, non_blocking=True)
        y_true = y_true.to(device, non_blocking=True)

        # Forward pass
        if training:
            if mixed_precision:
                with autocast():
                    y_pred = model(x)
                    y_pred_ups = upsample_pred_if_needed(y_pred, y_true)
                    loss = get_loss(y_pred_ups, y_true, criterion, aux_loss_wt=False)
            else:
                y_pred = model(x)
                y_pred_ups = upsample_pred_if_needed(y_pred, y_true)
                loss = get_loss(y_pred_ups, y_true, criterion, aux_loss_wt=False)
        else:
            with torch.no_grad():
                if mixed_precision:
                    with autocast():
                        y_pred = model(x)
                        y_pred_ups = upsample_pred_if_needed(y_pred, y_true)
                        loss = get_loss(y_pred_ups, y_true, criterion, aux_loss_wt=False)
                else:
                    y_pred = model(x)
                    y_pred_ups = upsample_pred_if_needed(y_pred, y_true)
                    loss = get_loss(y_pred_ups, y_true, criterion, aux_loss_wt=False)

        # Backward pass
        if training:
            if mixed_precision:
                model.module.scaler.scale(loss).backward()
                model.module.scaler.step(optimizer)
                model.module.scaler.update()
            else:
                loss.backward()
                optimizer.step()
            if scheduler:
                scheduler.step()

        meter_loss.update(loss, x.size(0))

        # Loss intra-epoch printing
        if (i_batch + 1) % log_freq == 0:

            print(f"{train_or_test.upper(): >5} [{i_batch + 1:04d}/{len(dataloader):04d}]"
                  f"\t\tLOSS: {meter_loss.running_average:.7f}")

            if training:
                wandb.log({"batch": len(dataloader) * epoch + i_batch, f"loss_{train_or_test}": loss})

    print(f"{train_or_test.upper(): >5} Complete!"
          f"\t\t\tLOSS: {meter_loss.avg:.7f}")

    loss = float(meter_loss.avg.detach().cpu().numpy())
    wandb.log({'epoch': epoch, f'loss_{train_or_test}': loss})

    return loss


def save_model(state, save_path, test_metric, best_metric, cfg, last_save_path, lowest_best=True, final=False):
    save = cfg['output']['save']
    if save == 'all' or final:
        torch.save(state, save_path)
    elif (test_metric < best_metric) == lowest_best:
        print(f"{test_metric:.5f} better than {best_metric:.5f} -> SAVING")
        if save == 'best':  # Delete previous best if using best only; otherwise keep previous best
            if last_save_path:
                try:
                    os.remove(last_save_path)
                except FileNotFoundError:
                    print(f"Failed to find {last_save_path}")
        best_metric = test_metric
        torch.save(state, save_path)
        last_save_path = save_path
    else:
        print(f"{test_metric:.5g} not improved from {best_metric:.5f}")
    return best_metric, last_save_path
