from pathlib import Path
import hashlib

IMAGE_DIR = Path("dataset/Images")
MASK_DIR = Path("dataset/Marks")

images = sorted(IMAGE_DIR.glob("*"))
masks = sorted(MASK_DIR.glob("*"))

print("Images:", len(images))
print("Masks :", len(masks))

image_names = {x.name for x in images}
mask_names = {x.name for x in masks}

missing_masks = image_names - mask_names
missing_images = mask_names - image_names

print("\nMissing masks:")
for x in sorted(missing_masks):
    print(x)

print("\nMissing images:")
for x in sorted(missing_images):
    print(x)

if not missing_masks and not missing_images:
    print("\nSUCCESS: Every image has a matching mask.")
else:
    print("\nWARNING: Dataset has unmatched files.")


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


print("\nDataset files:")
for image in images:
    mask = MASK_DIR / image.name

    if mask.exists():
        print(
            image.name,
            "IMAGE_HASH:", file_hash(image)[:12],
            "MASK_HASH:", file_hash(mask)[:12]
        )