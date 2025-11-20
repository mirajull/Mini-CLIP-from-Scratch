import os
from PIL import Image, ImageDraw

OUT_DIR = "data/images"
os.makedirs(OUT_DIR, exist_ok=True)

IMG_SIZE = 256
BG_COLOR = (255, 255, 255)

def make_square(path, color):
    img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), BG_COLOR)
    draw = ImageDraw.Draw(img)
    margin = 60
    draw.rectangle(
        [margin, margin, IMG_SIZE - margin, IMG_SIZE - margin],
        fill=color,
    )
    img.save(path)

def make_circle(path, color):
    img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), BG_COLOR)
    draw = ImageDraw.Draw(img)
    margin = 60
    draw.ellipse(
        [margin, margin, IMG_SIZE - margin, IMG_SIZE - margin],
        fill=color,
    )
    img.save(path)

if __name__ == "__main__":
    make_square(os.path.join(OUT_DIR, "red_square.png"), (255, 0, 0))
    make_square(os.path.join(OUT_DIR, "green_square.png"), (0, 255, 0))
    make_square(os.path.join(OUT_DIR, "blue_square.png"), (0, 0, 255))

    make_circle(os.path.join(OUT_DIR, "blue_circle.png"), (0, 0, 255))
    make_circle(os.path.join(OUT_DIR, "yellow_circle.png"), (255, 255, 0))
    make_circle(os.path.join(OUT_DIR, "red_circle.png"), (255, 0, 0))

    print("Toy images written to", OUT_DIR)
