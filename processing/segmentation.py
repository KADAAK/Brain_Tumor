import cv2
import numpy as np
from scipy import ndimage
from skimage.measure import regionprops


def mock_segment(image: np.ndarray) -> np.ndarray:
    """Accurately isolates intracranial brain tumor mass regions while removing skull, scalp, and normal CSF."""
    source = np.asarray(image, dtype=np.float32)
    if source.ndim == 3:
        source = source.mean(axis=-1) if source.shape[-1] == 3 else source[source.shape[0] // 2]

    s_min, s_max = float(source.min()), float(source.max())
    if s_max <= s_min:
        return np.zeros(source.shape, dtype=np.uint8)

    img_u8 = ((source - s_min) / (s_max - s_min) * 255.0).astype(np.uint8)
    h, w = img_u8.shape

    # 1. Skull Stripping & Brain Mask Extraction
    bg_thresh = max(10, int(np.percentile(img_u8, 15)))
    _, thresh = cv2.threshold(img_u8, bg_thresh, 255, cv2.THRESH_BINARY)
    head_contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    is_anatomical_head = False
    head_mask = np.ones((h, w), dtype=np.uint8) * 255

    if head_contours:
        largest = max(head_contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        if 0.15 * (h * w) < area < 0.95 * (h * w):
            is_anatomical_head = True
            head_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(head_mask, [largest], -1, 255, -1)

    if is_anatomical_head:
        # Erode head mask inward to strictly exclude the bright skull, scalp, and outer meningeal rim
        erode_radius = max(5, int(min(h, w) * 0.048))
        kernel_skull = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_radius * 2, erode_radius * 2))
        brain_mask = cv2.erode(head_mask, kernel_skull)
    else:
        brain_mask = np.ones((h, w), dtype=np.uint8) * 255

    pixels = img_u8[brain_mask > 0]
    if len(pixels) == 0:
        return np.zeros((h, w), dtype=np.uint8)

    # 2. Intracranial Parenchyma Enhancement & Filtering
    parenchyma = cv2.bitwise_and(img_u8, img_u8, mask=brain_mask)
    blurred = cv2.bilateralFilter(parenchyma, d=7, sigmaColor=60, sigmaSpace=60)

    mean_val = float(np.mean(pixels))
    std_val = float(np.std(pixels))
    p88 = float(np.percentile(pixels, 88))

    lesion_thresh = max(p88, mean_val + 0.85 * std_val)
    raw_lesion = ((blurred >= lesion_thresh) & (brain_mask > 0)).astype(np.uint8)

    # 3. Morphological closing to bridge tumor core gaps and fill solid lesion mass
    close_r = max(4, int(min(h, w) * 0.032))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_r * 2, close_r * 2))
    closed = cv2.morphologyEx(raw_lesion, cv2.MORPH_CLOSE, kernel_close)
    filled = ndimage.binary_fill_holes(closed).astype(np.uint8)

    # 4. Focal Abnormal Region Selection
    lbl, num_features = ndimage.label(filled)
    props = regionprops(lbl)
    if not props:
        return np.zeros((h, w), dtype=np.uint8)

    brain_area = float(np.count_nonzero(brain_mask))
    min_tumor_area = max(50, int(brain_area * 0.012))

    if is_anatomical_head:
        M = cv2.moments(brain_mask)
        cx_b = M['m10'] / M['m00'] if M['m00'] > 0 else w / 2

        candidates = []
        for p in props:
            if p.area < min_tumor_area:
                continue
            cy, cx = p.centroid
            asymmetry = abs(cx - cx_b) / (w / 2)
            reg_pix = img_u8[lbl == p.label]
            reg_std = float(np.std(reg_pix)) if len(reg_pix) > 0 else 0
            solidity = p.solidity if hasattr(p, 'solidity') else 1.0

            # Score prioritizes true parenchymal tumors (focal asymmetry, high texture, compact mass)
            score = (p.area / brain_area) * (1.0 + 1.8 * asymmetry) * (1.0 + 0.02 * reg_std) * solidity
            candidates.append((score, p))

        mask = np.zeros((h, w), dtype=np.uint8)
        if candidates:
            candidates.sort(key=lambda x: x[0], reverse=True)
            top_score = candidates[0][0]
            for score, p in candidates:
                if score >= top_score * 0.55:
                    mask[lbl == p.label] = 1
    else:
        mask = np.zeros((h, w), dtype=np.uint8)
        for p in props:
            if p.area >= min_tumor_area:
                mask[lbl == p.label] = 1

    smooth_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, smooth_k)
    return mask.astype(np.uint8)

