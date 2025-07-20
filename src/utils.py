import os
import random
import numpy as np
import torch
from typing import Dict, List
from sklearn.metrics import accuracy_score, classification_report


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_batch_accuracies(type_logits, loc_logits, batch):
    type_pred = type_logits.argmax(dim=1)
    loc_pred = loc_logits.argmax(dim=1)
    type_acc = (type_pred == batch['type_label']).float().mean().item()
    loc_acc = (loc_pred == batch['location_label']).float().mean().item()
    return type_acc, loc_acc


def aggregate_epoch_metrics(all_metrics):
    out = {}
    for k in all_metrics[0].keys():
        out[k] = float(np.mean([m[k] for m in all_metrics]))
    return out


def save_checkpoint(state: dict, is_best: bool, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    path_last = os.path.join(out_dir, 'last.pt')
    torch.save(state, path_last)
    if is_best:
        torch.save(state, os.path.join(out_dir, 'best.pt'))


def classification_reports(model, loader, device, type_inverse, loc_inverse):
    model.eval()
    type_trues, type_preds = [], []
    loc_trues, loc_preds = [], []
    with torch.no_grad():
        for batch in loader:
            for k in batch: batch[k] = batch[k].to(device)
            tl, ll, _, _ = model(batch['image'])
            type_trues.append(batch['type_label'].cpu())
            loc_trues.append(batch['location_label'].cpu())
            type_preds.append(tl.argmax(1).cpu())
            loc_preds.append(ll.argmax(1).cpu())
    type_trues = torch.cat(type_trues).numpy()
    loc_trues = torch.cat(loc_trues).numpy()
    type_preds = torch.cat(type_preds).numpy()
    loc_preds = torch.cat(loc_preds).numpy()
    rep_type = classification_report(type_trues, type_preds, target_names=[type_inverse[i] for i in sorted(type_inverse.keys())])
    rep_loc = classification_report(loc_trues, loc_preds, target_names=[loc_inverse[i] for i in sorted(loc_inverse.keys())])
    return rep_type, rep_loc