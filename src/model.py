import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBackbone(nn.Module):
    """A small CNN backbone. Replace with a stronger one (e.g. ResNet) for better results."""
    def __init__(self, in_ch=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, 3, stride=1, padding=1), nn.ReLU(),
        )
        self.out_channels = 256
    def forward(self, x):
        return self.features(x)

class SharedMLP(nn.Module):
    def __init__(self, in_ch, hidden=512):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_ch, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
    def forward(self, x):
        return self.fc(x)

class MultiTaskOutputs(nn.Module):
    def __init__(self, hidden, n_type, n_loc):
        super().__init__()
        self.cls_type = nn.Linear(hidden, n_type)
        self.cls_loc = nn.Linear(hidden, n_loc)
        self.bbox = nn.Linear(hidden, 4)
    def forward(self, h):
        return self.cls_type(h), self.cls_loc(h), self.bbox(h)

class MaskHead(nn.Module):
    def __init__(self, in_ch, out_ch=1):
        super().__init__()
        self.head = nn.Sequential(
            nn.ConvTranspose2d(in_ch, 128, 2, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 2, stride=2), nn.ReLU(),
            nn.Conv2d(64, out_ch, 1)
        )
    def forward(self, feat):
        return self.head(feat)

class DentalMultiHeadModel(nn.Module):
    def __init__(self, num_classes_type=4, num_classes_location=4):
        super().__init__()
        self.backbone = ConvBackbone()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.shared = SharedMLP(self.backbone.out_channels)
        self.outs = MultiTaskOutputs(512, num_classes_type, num_classes_location)
        self.mask_head = MaskHead(self.backbone.out_channels)
    def forward(self, x):
        feat = self.backbone(x)
        pooled = self.pool(feat).flatten(1)
        h = self.shared(pooled)
        type_logits, loc_logits, bbox_pred = self.outs(h)
        mask_logits = self.mask_head(feat)
        return type_logits, loc_logits, bbox_pred, mask_logits

# Loss computation helper
class LossComputer:
    def __init__(self, lambda_bbox=1.0, lambda_mask=1.0):
        self.lambda_bbox = lambda_bbox
        self.lambda_mask = lambda_mask
    def __call__(self, outputs, batch):
        type_logits, loc_logits, bbox_pred, mask_logits = outputs
        type_loss = F.cross_entropy(type_logits, batch['type_label'])
        loc_loss = F.cross_entropy(loc_logits, batch['location_label'])
        bbox_loss = F.smooth_l1_loss(bbox_pred, batch['bbox']) * self.lambda_bbox
        mask_loss = F.binary_cross_entropy_with_logits(mask_logits, batch['mask']) * self.lambda_mask
        total = type_loss + loc_loss + bbox_loss + mask_loss
        metrics = {
            'loss_total': total.item(),
            'loss_type': type_loss.item(),
            'loss_location': loc_loss.item(),
            'loss_bbox': bbox_loss.item(),
            'loss_mask': mask_loss.item(),
        }
        return total, metrics