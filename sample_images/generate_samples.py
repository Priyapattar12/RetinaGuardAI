"""
Generates a handful of SYNTHETIC fundus-like demo images so the app has
something to show in the gallery out of the box, with no internet access
and no real patient data required.

These are NOT real medical images -- they are simple procedurally drawn
circles with random "lesion" dots, built only so you can click through the
app's full flow (gallery -> analyze -> result) immediately.

For a real project, replace these with real de-identified fundus images
from a public dataset such as APTOS 2019 (Kaggle) or IDRiD, respecting
that dataset's license and terms of use.

Run with: python generate_samples.py
"""

import os
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

random.seed(42)
np.random.seed(42)

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
SIZE = 512


def base_fundus_canvas():
    cx, cy, r = SIZE // 2, SIZE // 2, SIZE // 2 - 10

    # Smooth radial warm gradient computed per-pixel (no banding/contour edges)
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / r
    dist = np.clip(dist, 0, 1)
    t = 1 - dist

    red = (120 + 80 * t).astype(np.uint8)
    green = (40 + 30 * t).astype(np.uint8)
    blue = (20 + 10 * t).astype(np.uint8)
    arr = np.stack([red, green, blue], axis=-1)

    # mask outside the circle to near-black
    outside = dist >= 1
    arr[outside] = (8, 4, 4)

    img = Image.fromarray(arr, mode="RGB")
    img = img.filter(ImageFilter.GaussianBlur(6))

    draw = ImageDraw.Draw(img)

    # Optic disc (bright soft circle, slightly off-center)
    disc_x, disc_y = cx + r // 3, cy
    draw.ellipse(
        [disc_x - 32, disc_y - 32, disc_x + 32, disc_y + 32],
        fill=(220, 190, 145),
    )

    # Simple thin branching vessels (low contrast so they don't read as lesions)
    for _ in range(10):
        angle = random.uniform(0, 2 * np.pi)
        length = random.uniform(80, r - 30)
        x1, y1 = disc_x, disc_y
        x2 = int(x1 + length * np.cos(angle))
        y2 = int(y1 + length * np.sin(angle))
        draw.line([x1, y1, x2, y2], fill=(110, 45, 35), width=1)

    img = img.filter(ImageFilter.GaussianBlur(2.5))
    return img, cx, cy, r


def add_lesions(img, cx, cy, r, dark_count, bright_count):
    draw = ImageDraw.Draw(img)
    for _ in range(dark_count):
        angle = random.uniform(0, 2 * np.pi)
        dist = random.uniform(0, r - 40)
        x = int(cx + dist * np.cos(angle))
        y = int(cy + dist * np.sin(angle))
        rad = random.randint(3, 9)
        draw.ellipse([x - rad, y - rad, x + rad, y + rad], fill=(60, 10, 10))

    for _ in range(bright_count):
        angle = random.uniform(0, 2 * np.pi)
        dist = random.uniform(0, r - 40)
        x = int(cx + dist * np.cos(angle))
        y = int(cy + dist * np.sin(angle))
        rad = random.randint(3, 8)
        draw.ellipse([x - rad, y - rad, x + rad, y + rad], fill=(240, 220, 160))

    return img.filter(ImageFilter.GaussianBlur(0.6))


SAMPLES = [
    ("sample_no_dr.jpg", 0, 0),
    ("sample_mild.jpg", 4, 2),
    ("sample_moderate.jpg", 12, 8),
    ("sample_severe.jpg", 28, 18),
    ("sample_proliferative.jpg", 50, 30),
]


def main():
    for filename, dark_n, bright_n in SAMPLES:
        img, cx, cy, r = base_fundus_canvas()
        img = add_lesions(img, cx, cy, r, dark_n, bright_n)
        path = os.path.join(OUT_DIR, filename)
        img.save(path, quality=92)
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
