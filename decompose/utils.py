import numpy as np
from PIL import Image
from pyora import Project


def images_to_ORA(images) -> Project:
    project = Project.new(*images[0].size)
    for i in range(len(images)):
        project.add_layer(images[i], "RGBA Layer " + str(i))
    return project


def palette_to_image(palette: list[list[int]]) -> Image.Image:
    palette_size = len(palette)
    print((palette_size))
    im = Image.new("RGB", (palette_size + 2, 3), (128, 128, 128))
    for i in range(palette_size):
        im.putpixel((i + 1, 1), tuple(palette[i]))
    im = im.resize((int(im.width * 32), int(im.height * 32)), Image.NEAREST)
    return im


def save_palette(palette: list[list[int]], image_path: str):
    palette_to_image(palette).save(image_path)


def invert_RGB8(color: tuple[int]) -> tuple:
    return (255 - color[0], 255 - color[1], 255 - color[2])


def fill_palette(p):
    len_colors = len(p)
    n = 7 - len_colors
    for i in range(n):
        p.append(p[-1])
