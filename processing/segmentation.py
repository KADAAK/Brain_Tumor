import cv2
import numpy as np
from scipy import ndimage
from skimage.measure import regionprops


def _segment_single_frame(source_u8: np.ndarray) -> np.ndarray:
    """Isolates tumor regions within a single brain MRI frame/view."""
    h, w = source_u8.shape
    if h < 25 or w < 25:
        return np.zeros((h, w), dtype=np.uint8)

    # 1. Skull Stripping & Brain Mask Extraction
    bg_thresh = max(10, int(np.percentile(source_u8, 15)))
    _, thresh = cv2.threshold(source_u8, bg_thresh, 255, cv2.THRESH_BINARY)
    head_contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    is_anatomical_head = False
    head_mask = np.ones((h, w), dtype=np.uint8) * 255

    if head_contours:
        largest = max(head_contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        if 0.10 * (h * w) < area < 0.95 * (h * w):
            is_anatomical_head = True
            head_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(head_mask, [largest], -1, 255, -1)

    if is_anatomical_head:
        # Erode head mask inward to strictly exclude the bright skull, scalp, and outer meningeal rim
        erode_radius = max(4, int(min(h, w) * 0.045))
        kernel_skull = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_radius * 2, erode_radius * 2))
        brain_mask = cv2.erode(head_mask, kernel_skull)
    else:
        brain_mask = np.ones((h, w), dtype=np.uint8) * 255

    pixels = source_u8[brain_mask > 0]
    if len(pixels) == 0:
        return np.zeros((h, w), dtype=np.uint8)

    # 2. Intracranial Parenchyma Enhancement & Filtering
    parenchyma = cv2.bitwise_and(source_u8, source_u8, mask=brain_mask)
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
    min_tumor_area = max(40, int(brain_area * 0.012))

    if is_anatomical_head:
        M = cv2.moments(brain_mask)
        cx_b = M['m10'] / M['m00'] if M['m00'] > 0 else w / 2

        candidates = []
        for p in props:
            if p.area < min_tumor_area:
                continue
            cy, cx = p.centroid
            asymmetry = abs(cx - cx_b) / (w / 2)
            reg_pix = source_u8[lbl == p.label]
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
                if score >= top_score * 0.50:
                    mask[lbl == p.label] = 1
    else:
        mask = np.zeros((h, w), dtype=np.uint8)
        for p in props:
            if p.area >= min_tumor_area:
                mask[lbl == p.label] = 1

    smooth_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, smooth_k)
    return mask.astype(np.uint8)


def _split_multiframe_panels(img: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Detects multi-frame MRI panels (grid sheets or multi-angle scans)."""
    h, w = img.shape[:2]
    row_means = img.mean(axis=1)
    col_means = img.mean(axis=0)

    row_gutter_thresh = max(12.0, float(np.percentile(row_means, 12)))
    col_gutter_thresh = max(12.0, float(np.percentile(col_means, 12)))

    is_row_dark = row_means <= row_gutter_thresh
    is_col_dark = col_means <= col_gutter_thresh

    def find_splits(is_dark, length):
        splits = [0]
        in_gutter = False
        gutter_start = 0
        for i, dark in enumerate(is_dark):
            if 0.08 * length < i < 0.92 * length:
                if dark and not in_gutter:
                    in_gutter = True
                    gutter_start = i
                elif not dark and in_gutter:
                    in_gutter = False
                    if i - gutter_start >= max(6, int(length * 0.015)):
                        splits.append((gutter_start + i) // 2)
        splits.append(length)
        return splits

    y_splits = find_splits(is_row_dark, h)
    x_splits = find_splits(is_col_dark, w)

    panels = []
    if len(y_splits) > 2 or len(x_splits) > 2:
        for i in range(len(y_splits) - 1):
            for j in range(len(x_splits) - 1):
                y1, y2 = y_splits[i], y_splits[i + 1]
                x1, x2 = x_splits[j], x_splits[j + 1]
                if (y2 - y1) >= 0.18 * h and (x2 - x1) >= 0.18 * w:
                    sub = img[y1:y2, x1:x2]
                    if sub.size > 0 and sub.mean() > 10:
                        panels.append((x1, y1, x2, y2))

    # Fallback to connected component head detection if grid gutters were irregular
    if len(panels) < 2:
        _, thresh = cv2.threshold(img, 15, 255, cv2.THRESH_BINARY)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, k)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        head_boxes = []
        for c in contours:
            area = cv2.contourArea(c)
            if 0.03 * (h * w) < area < 0.85 * (h * w):
                bx, by, bw, bh = cv2.boundingRect(c)
                pad_x, pad_y = max(8, int(bw * 0.08)), max(8, int(bh * 0.08))
                x1, y1 = max(0, bx - pad_x), max(0, by - pad_y)
                x2, y2 = min(w, bx + bw + pad_x), min(h, by + bh + pad_y)
                head_boxes.append((x1, y1, x2, y2))
        if len(head_boxes) >= 2:
            return sorted(head_boxes, key=lambda b: (b[1] // (h // 2), b[0]))
        return [(0, 0, w, h)]

    return panels


def mock_segment(image: np.ndarray) -> np.ndarray:
    """Accurately isolates brain tumor mass regions in single-frame, multi-frame, and multi-angle MRI scans."""
    source = np.asarray(image, dtype=np.float32)
    if source.ndim == 3:
        source = source.mean(axis=-1) if source.shape[-1] == 3 else source[source.shape[0] // 2]

    s_min, s_max = float(source.min()), float(source.max())
    if s_max <= s_min:
        return np.zeros(source.shape, dtype=np.uint8)

    img_u8 = ((source - s_min) / (s_max - s_min) * 255.0).astype(np.uint8)
    h, w = img_u8.shape

    panels = _split_multiframe_panels(img_u8)
    full_mask = np.zeros((h, w), dtype=np.uint8)

    if len(panels) >= 2:
        for x1, y1, x2, y2 in panels:
            sub = img_u8[y1:y2, x1:x2]
            sub_mask = _segment_single_frame(sub)
            full_mask[y1:y2, x1:x2] = np.maximum(full_mask[y1:y2, x1:x2], sub_mask)
    else:
        full_mask = _segment_single_frame(img_u8)

    return full_mask


