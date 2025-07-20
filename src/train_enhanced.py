#!/usr/bin/env python3
"""
Enhanced training script with learning rate scheduling and better logging
"""
import argparse
import os
import yaml
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
import matplotlib.pyplot as plt
import json

from dataset import DentalPatchDataset, collate_fn
from model import DentalMultiHeadModel, LossComputer
from utils import set_seed, compute_batch_accuracies, aggregate_epoch_metrics, save_checkpoint


class TrainingLogger:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.metrics_history = []
        
    def log_epoch(self, epoch, train_metrics, val_metrics, lr):
        entry = {
            'epoch': epoch,
            'learning_rate': lr,
            'train': train_metrics,
            'validation': val_metrics
        }
        self.metrics_history.append(entry)
        
        # Save metrics
        with open(os.path.join(self.output_dir, 'training_log.json'), 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
            
    def plot_metrics(self):
        if not self.metrics_history:
            return
            
        epochs = [e['epoch'] for e in self.metrics_history]
        train_loss = [e['train']['loss_total'] for e in self.metrics_history]
        val_loss = [e['validation']['loss_total'] for e in self.metrics_history]
        train_acc_type = [e['train']['acc_type'] for e in self.metrics_history]
        val_acc_type = [e['validation']['acc_type'] for e in self.metrics_history]
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        
        # Loss plots
        ax1.plot(epochs, train_loss, 'b-', label='Train')
        ax1.plot(epochs, val_loss, 'r-', label='Validation')
        ax1.set_title('Total Loss')
        ax1.legend()
        
        # Type accuracy plots
        ax2.plot(epochs, train_acc_type, 'b-', label='Train')
        ax2.plot(epochs, val_acc_type, 'r-', label='Validation')
        ax2.set_title('Type Classification Accuracy')
        ax2.legend()
        
        # Learning rate
        lrs = [e['learning_rate'] for e in self.metrics_history]
        ax3.plot(epochs, lrs, 'g-')
        ax3.set_title('Learning Rate')
        
        # Individual loss components
        train_type_loss = [e['train']['loss_type'] for e in self.metrics_history]
        train_bbox_loss = [e['train']['loss_bbox'] for e in self.metrics_history]
        ax4.plot(epochs, train_type_loss, label='Type')
        ax4.plot(epochs, train_bbox_loss, label='BBox')
        ax4.set_title('Training Loss Components')
        ax4.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'training_plots.png'))
        plt.close()


def create_scheduler(optimizer, cfg):
    scheduler_type = cfg.get('scheduler', {}).get('type', 'cosine')
    if scheduler_type == 'cosine':
        return CosineAnnealingLR(optimizer, T_max=cfg['train']['epochs'])
    elif scheduler_type == 'step':
        return StepLR(optimizer, step_size=cfg.get('scheduler', {}).get('step_size', 20), 
                     gamma=cfg.get('scheduler', {}).get('gamma', 0.1))
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/default.yaml')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--resume', default=None)
    parser.add_argument('--name', default='default', help='Experiment name')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Create experiment directory
    exp_dir = os.path.join(cfg['paths']['output_dir'], args.name)
    os.makedirs(exp_dir, exist_ok=True)
    
    # Save config
    with open(os.path.join(exp_dir, 'config.yaml'), 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False)
    
    set_seed(cfg['seed'])
    logger = TrainingLogger(exp_dir)
    
    # Data
    train_ds = DentalPatchDataset(cfg['paths']['train_csv'], cfg['paths']['train_dir'],
                                  image_size=cfg['image_size'], mask_size=cfg['mask_size'])
    valid_ds = DentalPatchDataset(cfg['paths']['valid_csv'], cfg['paths']['valid_dir'], 
                                  type_mapping=train_ds.type_mapping,
                                  image_size=cfg['image_size'], mask_size=cfg['mask_size'])
    
    train_loader = DataLoader(train_ds, batch_size=cfg['train']['batch_size'], shuffle=True,
                              num_workers=cfg['train']['num_workers'], pin_memory=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_ds, batch_size=cfg['train']['batch_size'], shuffle=False,
                              num_workers=cfg['train']['num_workers'], pin_memory=True, collate_fn=collate_fn)
    
    # Model
    model = DentalMultiHeadModel(cfg['num_classes_type'], cfg['num_classes_location']).to(args.device)
    loss_fn = LossComputer(lambda_bbox=cfg['train']['lambda_bbox'], lambda_mask=cfg['train']['lambda_mask'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['train']['lr'], weight_decay=cfg['train']['weight_decay'])
    scheduler = create_scheduler(optimizer, cfg)
    
    # Training loop
    best_val = float('inf')
    patience_counter = 0
    
    print(f"Starting training: {len(train_ds)} train, {len(valid_ds)} validation samples")
    
    for epoch in range(1, cfg['train']['epochs'] + 1):
        # Training
        model.train()
        train_metrics = []
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
        
        for batch in pbar:
            for k in batch: 
                batch[k] = batch[k].to(args.device)
            
            optimizer.zero_grad()
            outputs = model(batch['image'])
            loss, metrics = loss_fn(outputs, batch)
            
            type_acc, loc_acc = compute_batch_accuracies(outputs[0], outputs[1], batch)
            metrics['acc_type'] = type_acc
            metrics['acc_location'] = loc_acc
            
            loss.backward()
            optimizer.step()
            train_metrics.append(metrics)
            
            pbar.set_postfix({'loss': f"{metrics['loss_total']:.4f}", 
                            'acc': f"{type_acc:.3f}"})
        
        # Validation
        model.eval()
        val_metrics = []
        with torch.no_grad():
            for batch in tqdm(valid_loader, desc='Validation', leave=False):
                for k in batch: 
                    batch[k] = batch[k].to(args.device)
                outputs = model(batch['image'])
                _, metrics = loss_fn(outputs, batch)
                type_acc, loc_acc = compute_batch_accuracies(outputs[0], outputs[1], batch)
                metrics['acc_type'] = type_acc
                metrics['acc_location'] = loc_acc
                val_metrics.append(metrics)
        
        train_agg = aggregate_epoch_metrics(train_metrics)
        val_agg = aggregate_epoch_metrics(val_metrics)
        
        current_lr = optimizer.param_groups[0]['lr']
        logger.log_epoch(epoch, train_agg, val_agg, current_lr)
        
        print(f"Epoch {epoch}: Train Loss={train_agg['loss_total']:.4f} | "
              f"Val Loss={val_agg['loss_total']:.4f} | LR={current_lr:.2e}")
        
        # Early stopping
        if val_agg['loss_total'] < best_val:
            best_val = val_agg['loss_total']
            patience_counter = 0
            # Save best model
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_val': best_val,
                'config': cfg
            }, os.path.join(exp_dir, 'best.pt'))
        else:
            patience_counter += 1
            
        if patience_counter >= cfg['train']['early_stop_patience']:
            print("Early stopping!")
            break
            
        if scheduler:
            scheduler.step()
        
        # Plot every 10 epochs
        if epoch % 10 == 0:
            logger.plot_metrics()
    
    logger.plot_metrics()
    print(f"Training finished! Best validation loss: {best_val:.4f}")


if __name__ == '__main__':
    main()
