"""Train a standard U-Net. Run as: python -m ml.train --epochs 50"""
from __future__ import annotations
import argparse, json, random
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np, torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from ml.config import DEFAULT_CONFIG, TrainingConfig
from ml.dataset import PairedMRIDataset, split_pairs, validate_dataset
from ml.losses import build_loss
from ml.metrics import segmentation_metrics
from ml.models.unet import UNet

def seed_everything(seed:int)->None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False
def run_epoch(model,loader,criterion,device,optimizer=None,threshold=.5):
    training=optimizer is not None; model.train(training); totals={"loss":0.,"dice":0.,"iou":0.,"precision":0.,"recall":0.}; count=0
    with torch.set_grad_enabled(training):
        for batch in loader:
            x,y=batch["image"].to(device),batch["mask"].to(device); logits=model(x); loss=criterion(logits,y)
            if training: optimizer.zero_grad(); loss.backward(); optimizer.step()
            metrics=segmentation_metrics(logits,y,threshold); totals["loss"]+=loss.item(); count+=1
            for key,value in metrics.items(): totals[key]+=value
    return {k:v/max(count,1) for k,v in totals.items()}
def save_graph(history:dict,output:Path,key:str,label:str)->None:
    plt.figure(); plt.plot(history["epoch"],history[f"train_{key}"],label="Train"); plt.plot(history["epoch"],history[f"val_{key}"],label="Validation"); plt.xlabel("Epoch"); plt.ylabel(label); plt.legend(); plt.grid(); plt.tight_layout(); plt.savefig(output); plt.close()
def main()->None:
    parser=argparse.ArgumentParser(); parser.add_argument("--dataset-dir",type=Path); parser.add_argument("--epochs",type=int); parser.add_argument("--batch-size",type=int); parser.add_argument("--loss",choices=["dice","bce","bce_dice"]); args=parser.parse_args()
    config=TrainingConfig()
    for key in ("dataset_dir","epochs","batch_size","loss"):
        value=getattr(args,key); setattr(config, "loss_name" if key=="loss" else key, value) if value is not None else None
    seed_everything(config.seed); config.output_dir.mkdir(parents=True,exist_ok=True); config.checkpoint_dir.mkdir(parents=True,exist_ok=True)
    pairs,report=validate_dataset(config); splits=split_pairs(pairs,config); print(f"Validated {report['valid_pairs']} paired images; split: train={len(splits['train'])}, validation={len(splits['validation'])}, test={len(splits['test'])}")
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); print(f"Device: {device}")
    loaders={"train":DataLoader(PairedMRIDataset(splits["train"],config,True),batch_size=config.batch_size,shuffle=True,num_workers=config.num_workers),"validation":DataLoader(PairedMRIDataset(splits["validation"],config),batch_size=config.batch_size,shuffle=False,num_workers=config.num_workers)}
    model=UNet(config.in_channels,config.out_channels,config.base_channels).to(device); criterion=build_loss(config.loss_name); optimizer=AdamW(model.parameters(),lr=config.learning_rate,weight_decay=config.weight_decay); scheduler=ReduceLROnPlateau(optimizer,mode="max",patience=3,factor=.5)
    history={"epoch":[],"train_loss":[],"val_loss":[],"train_dice":[],"val_dice":[],"train_iou":[],"val_iou":[]}; best=-1.; stale=0; best_path=config.checkpoint_dir/"brain_tumor_unet_best.pth"
    for epoch in range(1,config.epochs+1):
        train=run_epoch(model,loaders["train"],criterion,device,optimizer,config.threshold); val=run_epoch(model,loaders["validation"],criterion,device,threshold=config.threshold); scheduler.step(val["dice"])
        print(f"Epoch {epoch}/{config.epochs}\nTrain Loss: {train['loss']:.4f}\nVal Loss: {val['loss']:.4f}\nVal Dice: {val['dice']:.4f}\nVal IoU: {val['iou']:.4f}")
        history["epoch"].append(epoch)
        for key in ("loss","dice","iou"): history[f"train_{key}"].append(train[key]); history[f"val_{key}"].append(val[key])
        if val["dice"]>best:
            best=val["dice"]; stale=0; torch.save({"model_state_dict":model.state_dict(),"config":config.as_dict(),"epoch":epoch,"val_metrics":val},best_path); print(f"Saved best model: {best_path}")
        else:
            stale+=1
            if stale>=config.early_stopping_patience: print("Early stopping triggered."); break
    (config.output_dir/"training_history.json").write_text(json.dumps(history,indent=2),encoding="utf-8"); (config.output_dir/"training_config.json").write_text(json.dumps(config.as_dict(),indent=2),encoding="utf-8")
    for key,label in (("loss","Loss"),("dice","Dice Score"),("iou","IoU")): save_graph(history,config.output_dir/f"{key}_graph.png",key,label)
    print(f"Training complete. Best validation Dice: {best:.4f}. Run: python -m ml.test --checkpoint {best_path}")
if __name__=="__main__": main()
