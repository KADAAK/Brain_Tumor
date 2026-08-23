"""CLI prediction: python -m ml.predict --image path/to/image.png"""
import argparse,json
from pathlib import Path
import cv2,numpy as np
from PIL import Image
from ml.config import TrainingConfig
from ml.inference.predictor import UNetPredictor
from ml.postprocessing import analyze_mask
def main():
 p=argparse.ArgumentParser();p.add_argument("--image",required=True,type=Path);p.add_argument("--checkpoint",type=Path,default=TrainingConfig().checkpoint_dir/"brain_tumor_unet_best.pth");p.add_argument("--output-dir",type=Path,default=TrainingConfig().output_dir/"predictions");args=p.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
 image=np.asarray(Image.open(args.image).convert("L"));mask=UNetPredictor(args.checkpoint).predict(image);stem=args.image.stem;Image.fromarray((mask*255).astype(np.uint8)).save(args.output_dir/f"{stem}_mask.png")
 base=cv2.cvtColor(image,cv2.COLOR_GRAY2RGB); contours,_=cv2.findContours(mask.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(base,contours,-1,(255,0,0),2);Image.fromarray(base).save(args.output_dir/f"{stem}_overlay.png")
 analysis=analyze_mask(mask);(args.output_dir/f"{stem}_analysis.json").write_text(json.dumps(analysis,indent=2),encoding="utf-8");print(json.dumps({"mask":str(args.output_dir/f'{stem}_mask.png'),"overlay":str(args.output_dir/f'{stem}_overlay.png'),**analysis},indent=2))
if __name__=="__main__":main()
