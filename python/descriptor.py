import copy
import cv2 as cv
import math
import numpy as np

from matplotlib import pyplot as plt
from scipy import misc

from keras.models import Model

from auxiliar import sliding_window

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


def thresholded(center, pixels):
    out = []
    for a in pixels:
        if a >= center:
            out.append(1)
        else:
            out.append(0)
    return out


class Descriptor:
    def __init__(self):
        print('Descriptor')

    @classmethod
    def get_hog(self, image):
        winSize = (128, 128)
        blockSize = (16, 16)
        blockStride = (8, 8)
        cellSize = (8, 8)
        nbins = 9
        derivAperture = 1
        winSigma = 4.
        histogramNormType = 0
        L2HysThreshold = 0.4
        gammaCorrection = 0
        nlevels = 64

        hog_desc = cv.HOGDescriptor(winSize,
                                    blockSize,
                                    blockStride,
                                    cellSize,
                                    nbins,
                                    derivAperture,
                                    winSigma,
                                    histogramNormType,
                                    L2HysThreshold,
                                    gammaCorrection,
                                    nlevels)
        desc = hog_desc.compute(image)
        feat = [item for sublist in desc for item in sublist]
        return feat

    @classmethod
    def get_clbp(self, image, P=8, R=5, S=8, W=16):
        hist = []
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        lbp_img = np.copy(image)

        for x in range(0, len(image)):
            for y in range(0, len(image[0])):
                center = image[x, y]
                pixels = []
                for point in range(0, P):
                    r = x + R * math.cos(2 * math.pi * point / P)
                    c = y - R * math.sin(2 * math.pi * point / P)
                    if r < 0 or c < 0:
                        pixels.append(0)
                        continue
                    if int(r) == r:
                        if int(c) != c:
                            c1 = int(c)
                            c2 = math.ceil(c)
                            w1 = (c2 - c) / (c2 - c1)
                            w2 = (c - c1) / (c2 - c1)

                            pixels.append(int((w1 * get_pixel_else_0(image, int(r), int(c)) + \
                                               w2 * get_pixel_else_0(image, int(r), math.ceil(c))) / (w1 + w2)))
                        else:
                            pixels.append(get_pixel_else_0(image, int(r), int(c)))
                    elif int(c) == c:
                        r1 = int(r)
                        r2 = math.ceil(r)
                        w1 = (r2 - r) / (r2 - r1)
                        w2 = (r - r1) / (r2 - r1)
                        pixels.append((w1 * get_pixel_else_0(image, int(r), int(c)) + \
                                       w2 * get_pixel_else_0(image, math.ceil(r), int(c))) / (w1 + w2))
                    else:
                        pixels.append(bilinear_interpolation(r, c, image))

                values = thresholded(center, pixels)
                res = 0
                for a in range(0, len(values)):
                    res += values[a] * (2 ** a)

                lbp_img.itemset((x, y), res)

        print lbp_img
        crop_img = lbp_img[R:lbp_img.shape[0]-R, R:lbp_img.shape[1]-R]
        for (row, col, window) in sliding_window(crop_img, window_size=(W,W), step_size=S):
            window_hist, bins = np.histogram(window.flatten(), 256, [0, 256])
            hist.append(window_hist)
        return hist

    @classmethod
    def get_uniform_clbp(self, image, P=8, R=5):
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        lbp_img = np.copy(image)

        unassigned = []
        pixel_values = set()
        variating_blocks = 0

        for x in range(0, len(image)):
            for y in range(0, len(image[0])):
                center = image[x, y]
                pixels = []
                for point in range(1, P + 1):
                    r = x + R * math.cos(2 * math.pi * point / P)
                    c = y - R * math.sin(2 * math.pi * point / P)
                    if r < 0 or c < 0:
                        pixels.append(0)
                        continue
                    if int(r) == r:
                        if int(c) != c:
                            c1 = int(c)
                            c2 = math.ceil(c)
                            w1 = (c2 - c) / (c2 - c1)
                            w2 = (c - c1) / (c2 - c1)

                            pixels.append(int((w1 * get_pixel_else_0(image, int(r), int(c)) + \
                                               w2 * get_pixel_else_0(image, int(r), math.ceil(c))) / (w1 + w2)))
                        else:
                            pixels.append(get_pixel_else_0(image, int(r), int(c)))
                    elif int(c) == c:
                        r1 = int(r)
                        r2 = math.ceil(r)
                        w1 = (r2 - r) / (r2 - r1)
                        w2 = (r - r1) / (r2 - r1)
                        pixels.append((w1 * get_pixel_else_0(image, int(r), int(c)) + \
                                       w2 * get_pixel_else_0(image, math.ceil(r), int(c))) / (w1 + w2))
                    else:
                        pixels.append(bilinear_interpolation(r, c, image))

                values = thresholded(center, pixels)
                variations = find_variations(values)
                if variations <= 2:
                    res = 0
                    variating_blocks += 1
                    for a in range(0, len(values)):
                        res += values[a] * 2 ** a
                    lbp_img.itemset((x, y), res)
                    pixel_values.add(res)
                else:
                    unassigned.append((x, y))
            # print x

        unassigned_value = len(pixel_values)
        pixel_values = sorted(pixel_values)
        no_of_pixel_values = len(pixel_values)
        trans_p1_u2 = {}
        for p in range(0, len(pixel_values)):
            trans_p1_u2[pixel_values[p]] = p

        for r in range(0, len(lbp_img)):
            for c in range(0, len(lbp_img[0])):
                if (r, c) in unassigned:
                    lbp_img.itemset((r, c), unassigned_value)
                else:
                    p1 = lbp_img[r, c]
                    lbp_img.itemset((r, c), trans_p1_u2[p1])

        hist, bins = np.histogram(lbp_img.flatten(), no_of_pixel_values + 1, [0, no_of_pixel_values])
        return hist

    @classmethod
    def get_deep_feature(self, image, vgg_model, layer_name='fc6'):
        im = misc.imresize(image, (224, 224)).astype(np.float32)
        aux = copy.copy(im)
        im[:, :, 0] = aux[:, :, 2]
        im[:, :, 2] = aux[:, :, 0]

        # Remove image mean
        im[:, :, 0] -= 93.5940
        im[:, :, 1] -= 104.7624
        im[:, :, 2] -= 129.1863
        im = np.transpose(im, (2, 0, 1))  # If using Theano
        image = np.expand_dims(im, axis=0)

        intermediate_layer = Model(input=vgg_model.input, output=vgg_model.get_layer(layer_name).output)
        intermediate_output = intermediate_layer.predict(image)
        return intermediate_output
