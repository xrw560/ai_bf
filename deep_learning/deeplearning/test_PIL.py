# -*- coding: utf-8 -*-

from PIL import Image, ImageDraw
import numpy as np


def draw1():
    arr = []
    for _ in range(20):
        row = []
        for _ in range(10):
            row.append([255, 255, 0])
        arr.append(row)

    arr = np.array(arr)
    arr = np.uint8(arr)
    # print(arr)
    print(len(arr[0]), len(arr))
    img = Image.fromarray(arr, 'RGB')
    img.show()


def draw2():
    img = Image.new("RGB", (20, 20), (0, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.point((10, 10), fill=(255, 0, 0))
    draw.rectangle((5, 5, 15, 20), outline=(0, 0, 0), fill=(255, 0, 0))
    draw.ellipse((0, 0, 20, 10), outline=(0, 0, 255))
    img.show()


if __name__ == "__main__":
    draw2()
