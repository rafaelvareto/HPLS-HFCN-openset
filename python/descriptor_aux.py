import cv2 as cv
import math


def bilinear_interpolation(x, y, img):
    r = x
    c = y
    x1, y1 = int(r), int(c)
    x2, y2 = math.ceil(r), math.ceil(c)
    r1 = (x2 - x) / (x2 - x1) * get_pixel_else_0(img, x1, y1) + (x - x1) / (x2 - x1) * get_pixel_else_0(img, x2, y1)
    r2 = (x2 - x) / (x2 - x1) * get_pixel_else_0(img, x1, y2) + (x - x1) / (x2 - x1) * get_pixel_else_0(img, x2, y2)
    return (y2 - y) / (y2 - y1) * r1 + (y - y1) / (y2 - y1) * r2


def find_variations(pixel_values):
    prev = pixel_values[-1]
    t = 0
    for p in range(0, len(pixel_values)):
        cur = pixel_values[p]
        if cur != prev:
            t += 1
        prev = cur
    return t


def get_pixel_else_0(l, idx, idy):
    if idx < int(len(l)) - 1 and idy < len(l[0]):
        return l[int(idx), int(idy)]
    else:
        return 0


def load_images(path, image_list, display=False):
    for name in image_list:
        image = cv.imread(path + '/' + name, cv.IMREAD_COLOR)
        if display:
            cv.imshow('img', image)
            cv.waitKey(20)
    if display:
        cv.destroyAllWindows()


def load_txt_file(file_name):
    my_file = open(file_name, 'r')
    my_list = []
    for line in my_file:
        line = line.rstrip()
        my_list.append(line)
    return my_list


def thresholded(center, pixels):
    out = []
    for a in pixels:
        if a >= center:
            out.append(1)
        else:
            out.append(0)
    return out
