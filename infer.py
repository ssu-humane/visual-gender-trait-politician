# -*- encoding: utf-8 -*-
import os, sys
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from pathlib import Path
import argparse


# class Collator(object):
#     def __init__(self, device):
#         self.df = df
#         self.device = device

#     def __call__(self, batch):
#         (b_images, b_labels) = zip(*batch)

#         b_images = torch.stack(b_images).to(self.device, dtype=torch.float)
#         b_labels = torch.tensor(b_labels).to(self.device, dtype=torch.float)

#         return b_images, b_labels


class ElectionImageDatset(Dataset):

    def __init__(self, df, imgdir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.images = [Path(imgdir) / imgpath for imgpath in df["path"].to_list()]
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path)

        if self.transform:
            image = self.transform(image)

        return image
    

def infer(model_path, img_dir, infer_df, batch_size):
    model = models.resnet34(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(nn.Linear(num_ftrs, 14), nn.Sigmoid())
    model.load_state_dict(torch.load(model_path))

    device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")
    model = model.to(device)
    model.eval()
    
    transform_rgb = transforms.Lambda(lambda image: image.convert('RGB'))
    dataset_transform = transforms.Compose([
        transform_rgb,
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    infer_dataset = ElectionImageDatset(infer_df, img_dir, transform=dataset_transform)
    infer_dataset_size = len(infer_df.index)
    #collator = Collator("cpu")
    #infer_dataset_loader = DataLoader(infer_dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)
    infer_dataset_loader = DataLoader(infer_dataset, batch_size=batch_size, shuffle=False)

    preds_all = []; counter = 0
    for _, batch in enumerate(infer_dataset_loader):
        b_inputs = batch
        b_inputs = b_inputs.to(device, dtype=torch.float32)

        print(f"TRAIT INFERENCE: {counter}/{infer_dataset_size}")
        counter += batch_size
        
        with torch.no_grad():
            # forward
            outputs = model(b_inputs).squeeze()  # [batch, 14]
            preds_all.append(outputs)

    preds_all = torch.cat(preds_all, axis=0)  # [data size, 14]
    preds_all = preds_all.detach().cpu().numpy()

    return preds_all


def load_data(datapath, img_dir):
    # Expected header: path
    df = pd.read_csv(datapath)
    
    valid_idx = []
    for idx, row in df.iterrows():
        if os.path.exists(img_dir / row['path']):
            valid_idx.append(idx)

    target_df = df.iloc[valid_idx]
    print(f"Input data length: {len(df.index)} | Valid data length: {len(target_df.index)}")

    return target_df


def save_data(outpath, data_df, traits):
    column_names = ['Agreeable', 'Ambitious', 'Caring', 'Communal', 'Confident', 'Energetic', 'Feminine', 
                'Formal', 'Friendly', 'Masculine', 'Maternal','Patriotic', 'Professional', 'Qualified']
    print(traits.shape)
    result_df = pd.concat([data_df, pd.DataFrame(traits)], axis=1)
    result_df.columns = list(data_df.columns) + column_names
    result_df.to_csv(outpath, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Model-related params
    parser.add_argument("--model_path", default="model.pt", type=Path)
    parser.add_argument("--data_path", type=Path, required=True)
    parser.add_argument("--result_path", type=Path, required=True)
    parser.add_argument("--img_dir", type=Path, required=True)
    parser.add_argument("--batch_size", type=int, default=40)

    args = parser.parse_args()

    if not args.data_path.exists():
        print("ERR: Data do not exist", file=sys.stderr)
        exit()

    if not args.result_path.parent.exists():
        os.makedirs(args.result_path.parent)
    
    df = load_data(args.data_path, args.img_dir)
    inferred_traits = infer(args.model_path, args.img_dir, df, args.batch_size)
    save_data(args.result_path, df, inferred_traits)
