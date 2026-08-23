from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw
from backend.utils.image_utils import normalize_uint8


class VisualizationService:
    def generate(self, study_id: str, image, mask, components, output_dir: Path) -> dict:
        output_dir.mkdir(parents=True, exist_ok=True)
        base=normalize_uint8(image)
        original=output_dir / f"{study_id}_original.png"; Image.fromarray(base).save(original)
        segmentation=output_dir / f"{study_id}_mask.png"; Image.fromarray((mask.astype(np.uint8)*255)).save(segmentation)
        rgb=np.dstack([base,base,base]); contours,_=cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(rgb, contours, -1, (255,0,0), 2)
        annotated=Image.fromarray(rgb); draw=ImageDraw.Draw(annotated)
        crops=[]
        for item in components:
            y,x=item.centroid[-2:]; draw.text((x+4,y+4), item.tumor_id, fill=(255,0,0), stroke_width=1, stroke_fill=(255,255,255))
            minr,minc,maxr,maxc=item.bbox[-4:]; crop=annotated.crop((max(0,minc-8),max(0,minr-8),min(base.shape[1],maxc+8),min(base.shape[0],maxr+8)))
            cp=output_dir/f"{study_id}_{item.tumor_id}.png"; crop.save(cp); crops.append(cp)
        annotated_path=output_dir/f"{study_id}_annotated.png"; annotated.save(annotated_path)
        return {"original":original,"segmentation":segmentation,"annotated":annotated_path,"crops":crops}
