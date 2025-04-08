import io
from contextlib import redirect_stdout
from PIL import Image, ImageDraw, ImageFont

class ShowModel:
    def __init__(self, model, font_path="arial.ttf", font_size=14):
        # Get output of model.summary()
        with io.StringIO() as buf, redirect_stdout(buf):
            model.summary()
            output = buf.getvalue()

        # Write text on file
        with open('model.txt', 'a', encoding='utf-8') as f:
            f.write(output)
            f.write('\n\n ')

        # Create image
        self.text_to_image(output, font_path, font_size)

    def text_to_image(self, text, font_path, font_size, output_path="summary_image.png"):
        lines = text.split("\n")
        try:
            font = ImageFont.truetype(font_path, font_size)
        except OSError:
            font = ImageFont.load_default()

        width = int(max([font.getlength(line) for line in lines]) + 20)
        height = font_size * len(lines) + 20

        img = Image.new("RGB", (width, height), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)

        for i, line in enumerate(lines):
            draw.text((10, i * font_size), line, font=font, fill=(0, 0, 0))

        img.save(output_path)
        print(f"Saved image: {output_path}")
