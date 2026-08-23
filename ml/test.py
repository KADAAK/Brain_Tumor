"""Evaluate the best model once on the untouched test split and save comparisons."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np,torch
from torch.utils.data import DataLoader
from ml.config import TrainingConfig
from ml.dataset import PairedMRIDataset,split_pairs,validate_dataset
from ml.losses import build_loss
from ml.metrics import segmentation_metrics
from ml.models.unet import UNet
def main():
 p=argparse.ArgumentParser();p.add_argument("--checkpoint",type=Path,default=TrainingConfig().checkpoint_dir/"brain_tumor_unet_best.pth");p.add_argument("--dataset-dir",type=Path);args=p.parse_args(); config=TrainingConfig();config.dataset_dir=args.dataset_dir or config.dataset_dir
 pairs,_=validate_dataset(config); test_pairs=split_pairs(pairs,config)["test"]; checkpoint=torch.load(args.checkpoint,map_location="cpu",weights_only=False); saved=checkpoint.get("config",{})
 for n in ("image_size","in_channels","out_channels","base_channels","threshold","loss_name"):
  if n in saved:setattr(config,n,tuple(saved[n]) if n=="image_size" else saved[n])
 device=torch.device("cuda" if torch.cuda.is_available() else "cpu");model=UNet(config.in_channels,config.out_channels,config.base_channels).to(device);model.load_state_dict(checkpoint["model_state_dict"]);model.eval();loader=DataLoader(PairedMRIDataset(test_pairs,config),batch_size=1); totals={k:0. for k in ("loss","dice","iou","precision","recall")}; criterion=build_loss(config.loss_name); visual_dir=config.output_dir/"test_comparisons";visual_dir.mkdir(parents=True,exist_ok=True)
 with torch.no_grad():
  for index,batch in enumerate(loader):
   x,y=batch["image"].to(device),batch["mask"].to(device); logits=model(x); totals["loss"]+=criterion(logits,y).item(); metrics=segmentation_metrics(logits,y,config.threshold)
   for k,v in metrics.items():totals[k]+=v
   image=x[0,0].cpu().numpy();truth=y[0,0].cpu().numpy();pred=(torch.sigmoid(logits)[0,0].cpu().numpy()>=config.threshold);overlay=np.dstack([image,image,image]);overlay=(overlay-overlay.min())/max(overlay.max()-overlay.min(),1e-6);overlay[pred]=[1,0,0]
   fig,ax=plt.subplots(1,4,figsize=(14,4));
   for a,data,title,cmap in zip(ax,[image,truth,pred,overlay],["Original MRI","Ground Truth Mask","Predicted Mask","Overlay"],["gray","gray","gray",None]):a.imshow(data,cmap=cmap);a.set_title(title);a.axis("off")
   fig.tight_layout();fig.savefig(visual_dir/f"{Path(batch['name'][0]).stem}_comparison.png");plt.close(fig)
 results={k:v/max(len(loader),1) for k,v in totals.items()};(config.output_dir/"test_metrics.json").write_text(json.dumps(results,indent=2),encoding="utf-8");print("Test metrics:",json.dumps(results,indent=2));print(f"Saved comparisons to {visual_dir}")
if __name__=="__main__":main()
