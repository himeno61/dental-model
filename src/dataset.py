import os
from typing import Optional, Callable, Dict, Any
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

LOCATION_CLASSES = {"Upper-Left": 0, "Upper-Right": 1, "Lower-Left": 2, "Lower-Right": 3}
LOCATION_ID2NAME = {v: k for k, v in LOCATION_CLASSES.items()}

DEFAULT_TYPES = ["Fillings", "Implant", "Cavity", "Impacted Tooth"]

def build_type_mapping(df: pd.DataFrame, provided: Optional[list] = None):
    if provided:
        types = provided
    else:
        types = sorted(df['class'].unique())
    mapping = {c: i for i, c in enumerate(types)}
    inverse = {i: c for c, i in mapping.items()}
    return mapping, inverse

def derive_location_label(xmin: float, xmax: float, ymin: float, ymax: float, img_w: float, img_h: float) -> int:
    x_center = 0.5 * (xmin + xmax)
    y_center = 0.5 * (ymin + ymax)
    if y_center < img_h / 2 and x_center < img_w / 2:
        return LOCATION_CLASSES["Upper-Left"]
    elif y_center < img_h / 2 and x_center >= img_w / 2:
        return LOCATION_CLASSES["Upper-Right"]
    elif y_center >= img_h / 2 and x_center < img_w / 2:
        return LOCATION_CLASSES["Lower-Left"]
    else:
        return LOCATION_CLASSES["Lower-Right"]

class DentalPatchDataset(Dataset):
    """Dataset reading YOLO-like CSV: filename,width,height,class,xmin,ymin,xmax,ymax"""
    def __init__(self,
                 csv_file: str,
                 img_dir: str,
                 type_mapping: Dict[str, int] = None,
                 transform: Optional[Callable] = None,
                 image_size: int = 256,
                 mask_size: int = 28,
                 generate_rect_mask: bool = True):
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.image_size = image_size
        self.mask_size = mask_size
        self.generate_rect_mask = generate_rect_mask
        self.type_mapping, self.type_inverse = build_type_mapping(self.df, list(type_mapping.keys()) if type_mapping else DEFAULT_TYPES)
        # if user supplied mapping ensure order
        if type_mapping:
            self.type_mapping = type_mapping
            self.type_inverse = {v: k for k, v in type_mapping.items()}

        self.transform = transform or T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row['filename'])
        image = Image.open(img_path).convert('RGB')
        w_img, h_img = image.size

        xmin, ymin, xmax, ymax = float(row['xmin']), float(row['ymin']), float(row['xmax']), float(row['ymax'])
        type_name = row['class']
        type_id = self.type_mapping[type_name]
        loc_id = derive_location_label(xmin, xmax, ymin, ymax, w_img, h_img)

        # Normalize bbox to original size then to [0,1]
        bbox_norm = torch.tensor([
            xmin / w_img,
            ymin / h_img,
            xmax / w_img,
            ymax / h_img
        ], dtype=torch.float32)

        img_t = self.transform(image)

        # Generate rectangular mask (optionally refine later)
        if self.generate_rect_mask:
            # mask in original resolution of model's spatial feature? we keep simple fixed mask_size
            mask = torch.zeros((self.mask_size, self.mask_size), dtype=torch.float32)
            # map bbox to mask coordinates
            x0 = int(bbox_norm[0] * self.mask_size)
            y0 = int(bbox_norm[1] * self.mask_size)
            x1 = max(x0 + 1, int(bbox_norm[2] * self.mask_size))
            y1 = max(y0 + 1, int(bbox_norm[3] * self.mask_size))
            mask[y0:y1, x0:x1] = 1.0
            mask = mask.unsqueeze(0)
        else:
            mask = torch.zeros(1, self.mask_size, self.mask_size)

        sample = {
            'image': img_t,
            'bbox': bbox_norm,
            'type_label': torch.tensor(type_id, dtype=torch.long),
            'location_label': torch.tensor(loc_id, dtype=torch.long),
            'mask': mask
        }
        return sample

def collate_fn(batch):
    keys = batch[0].keys()
    out = {}
    for k in keys:
        out[k] = torch.stack([b[k] for b in batch])
    return out