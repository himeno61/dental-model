import argparse
import os
import yaml
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from dataset import DentalPatchDataset, collate_fn
from model import DentalMultiHeadModel, LossComputer
from utils import set_seed, compute_batch_accuracies, aggregate_epoch_metrics, save_checkpoint


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/default.yaml')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--resume', default=None, help='Path to checkpoint to resume')
    return ap.parse_args()


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def make_dataloaders(cfg, type_mapping=None):
    train_ds = DentalPatchDataset(cfg['paths']['train_csv'], cfg['paths']['train_dir'], type_mapping=type_mapping,
                                  image_size=cfg['image_size'], mask_size=cfg['mask_size'])
    valid_ds = DentalPatchDataset(cfg['paths']['valid_csv'], cfg['paths']['valid_dir'], type_mapping=train_ds.type_mapping,
                                  image_size=cfg['image_size'], mask_size=cfg['mask_size'])

    train_loader = DataLoader(train_ds, batch_size=cfg['train']['batch_size'], shuffle=True,
                              num_workers=cfg['train']['num_workers'], pin_memory=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_ds, batch_size=cfg['train']['batch_size'], shuffle=False,
                              num_workers=cfg['train']['num_workers'], pin_memory=True, collate_fn=collate_fn)
    return train_ds, valid_ds, train_loader, valid_loader


def train_one_epoch(model, loader, loss_fn, optimizer, device, log_interval=50):
    model.train()
    all_metrics = []
    for i, batch in enumerate(tqdm(loader, desc='Train', leave=False)):
        for k in batch: batch[k] = batch[k].to(device)
        optimizer.zero_grad()
        outputs = model(batch['image'])
        loss, metrics = loss_fn(outputs, batch)
        loss.backward()
        optimizer.step()
        # add accuracies
        type_acc, loc_acc = compute_batch_accuracies(outputs[0], outputs[1], batch)
        metrics['acc_type'] = type_acc
        metrics['acc_location'] = loc_acc
        all_metrics.append(metrics)
        if (i + 1) % log_interval == 0:
            avg = aggregate_epoch_metrics(all_metrics)
            print(f"Step {i+1}: loss={avg['loss_total']:.4f} type_acc={avg['acc_type']:.3f} loc_acc={avg['acc_location']:.3f}")
    return aggregate_epoch_metrics(all_metrics)


def evaluate(model, loader, loss_fn, device):
    model.eval()
    all_metrics = []
    with torch.no_grad():
        for batch in tqdm(loader, desc='Valid', leave=False):
            for k in batch: batch[k] = batch[k].to(device)
            outputs = model(batch['image'])
            _, metrics = loss_fn(outputs, batch)
            type_acc, loc_acc = compute_batch_accuracies(outputs[0], outputs[1], batch)
            metrics['acc_type'] = type_acc
            metrics['acc_location'] = loc_acc
            all_metrics.append(metrics)
    return aggregate_epoch_metrics(all_metrics)


def main():
    args = parse_args()
    cfg = load_config(args.config)
    os.makedirs(cfg['paths']['output_dir'], exist_ok=True)

    set_seed(cfg['seed'])

    train_ds, valid_ds, train_loader, valid_loader = make_dataloaders(cfg)

    model = DentalMultiHeadModel(cfg['num_classes_type'], cfg['num_classes_location']).to(args.device)
    loss_fn = LossComputer(lambda_bbox=cfg['train']['lambda_bbox'], lambda_mask=cfg['train']['lambda_mask'])
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['train']['lr'], weight_decay=cfg['train']['weight_decay'])

    start_epoch = 1
    best_val = float('inf')
    patience_counter = 0

    if args.resume:
        ckpt = torch.load(args.resume, map_location=args.device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt.get('epoch', 0) + 1
        best_val = ckpt.get('best_val', best_val)
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    for epoch in range(start_epoch, cfg['train']['epochs'] + 1):
        print(f"Epoch {epoch}/{cfg['train']['epochs']}")
        train_metrics = train_one_epoch(model, train_loader, loss_fn, optimizer, args.device, cfg['logging']['log_interval'])
        val_metrics = evaluate(model, valid_loader, loss_fn, args.device)

        print(f"Train: loss={train_metrics['loss_total']:.4f} type_acc={train_metrics['acc_type']:.3f} loc_acc={train_metrics['acc_location']:.3f}")
        print(f"Valid: loss={val_metrics['loss_total']:.4f} type_acc={val_metrics['acc_type']:.3f} loc_acc={val_metrics['acc_location']:.3f}")

        # Early stopping logic
        improved = val_metrics['loss_total'] < best_val
        if improved:
            best_val = val_metrics['loss_total']
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= cfg['train']['early_stop_patience']:
            print("Early stopping triggered.")
            break

        # Save checkpoint
        state = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_val': best_val,
            'config': cfg
        }
        save_checkpoint(state, improved, cfg['paths']['output_dir'])

    print("Training finished.")

if __name__ == '__main__':
    main()