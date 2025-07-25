from transformers import AutoModel, AutoConfig, AutoTokenizer
import struct
import numpy as np
from tqdm import tqdm
import os
import torch
import itertools
from config import CFG
from create_dataloaders import make_train_dfs, make_valid_dfs, build_loaders
from CLIP import CLIPModel
from loss_function import AvgMeter, get_lr




def train_epoch(model, train_loader, optimizer, lr_scheduler, step):
    loss_meter = AvgMeter()
    tqdm_object = tqdm(train_loader, total=len(train_loader))
    for batch in tqdm_object:
        batch = {k: v.to(CFG.device) for k, v in batch.items() if k != "caption"}
        loss = model(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step == "batch":
            lr_scheduler.step()

        count = batch["image"].size(0)
        loss_meter.update(loss.item(), count)

        tqdm_object.set_postfix(train_loss=loss_meter.avg, lr=get_lr(optimizer))
    return loss_meter


def valid_epoch(model, valid_loader):
    loss_meter = AvgMeter()

    tqdm_object = tqdm(valid_loader, total=len(valid_loader))
    for batch in tqdm_object:
        batch = {k: v.to(CFG.device) for k, v in batch.items() if k != "caption"}
        loss = model(batch)

        count = batch["image"].size(0)
        loss_meter.update(loss.item(), count)

        tqdm_object.set_postfix(valid_loss=loss_meter.avg)
    return loss_meter


def extract_features(data_loader, model, save_path="./fvec/fvecs.bin"):
    """
    Trích xuất đặc trưng từ mô hình và lưu vào file nhị phân.
    """
    device = CFG.device
    feature_vectors = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Extracting features"):
            images = batch["image"].to(device)
            features = model.image_encoder(images)  # Extract features using ImageEncoder
            feature_vectors.append(features)

    feature_vectors = torch.cat(feature_vectors, dim=0)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        fvecs = feature_vectors.cpu().numpy()
        fmt = f"{np.prod(fvecs.shape)}f"
        f.write(struct.pack(fmt, *(fvecs.flatten())))
    print(f"Feature vectors saved to {save_path}")


def main():
    train_df = make_train_dfs()
    valid_df = make_valid_dfs()
    # tokenizer = DistilBertTokenizer.from_pretrained(CFG.text_tokenizer)
    tokenizer = AutoTokenizer.from_pretrained(CFG.text_tokenizer)
    train_loader = build_loaders(train_df, tokenizer, mode="train")
    valid_loader = build_loaders(valid_df, tokenizer, mode="valid")

    model = CLIPModel().to(CFG.device)
    params = [
        {"params": model.image_encoder.parameters(), "lr": CFG.image_encoder_lr},
        {"params": model.text_encoder.parameters(), "lr": CFG.text_encoder_lr},
        {"params": itertools.chain(
            model.image_projection.parameters(), model.text_projection.parameters()
        ), "lr": CFG.head_lr, "weight_decay": CFG.weight_decay}
    ]
    optimizer = torch.optim.AdamW(params, weight_decay=0.)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=CFG.patience, factor=CFG.factor
    )
    step = "epoch"

    best_loss = float('inf')
    for epoch in range(CFG.epochs):
        print(f"Epoch: {epoch + 1}")
        model.train()
        train_loss = train_epoch(model, train_loader, optimizer, lr_scheduler, step)
        model.eval()
        with torch.no_grad():
            valid_loss = valid_epoch(model, valid_loader)

        if valid_loss.avg < best_loss:
            best_loss = valid_loss.avg
            torch.save(model.state_dict(), "best.pt")
            print("Saved Best Model!")

        lr_scheduler.step(valid_loss.avg)
    torch.save(model.image_encoder.state_dict(), "image_encoder_weights.pt")
    print("Image encoder weights saved!")

    extract_features(train_loader, model, save_path="fvec/fvecs_train.bin")
    extract_features(valid_loader, model, save_path="fvec/fvecs_valid.bin")