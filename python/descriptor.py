from matplotlib import pyplot as plt

import cv2 as cv
import math
import numpy as np

from descriptor_aux import bilinear_interpolation, find_variations, get_pixel_else_0, thresholded


class Descriptor:
    def __init__(self):
        print('Descriptor')

    @classmethod
    def get_hog(self, image):
        hog_desc = cv.HOGDescriptor()
        desc = hog_desc.compute(image)
        return desc

    @classmethod
    def get_clbp(self, image, P=8,R=1):
        transformed_img = np.copy(image)

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

                transformed_img.itemset((x, y), res)
            #print x

        hist, bins = np.histogram(transformed_img.flatten(), 256, [0, 256])
        return hist

    @classmethod
    def get_uniform_clbp(self, image, P=8,R=1):
        transformed_img = np.copy(image)

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
                    transformed_img.itemset((x, y), res)
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

        for r in range(0, len(transformed_img)):
            for c in range(0, len(transformed_img[0])):
                if (r, c) in unassigned:
                    transformed_img.itemset((r, c), unassigned_value)
                else:
                    p1 = transformed_img[r, c]
                    transformed_img.itemset((r, c), trans_p1_u2[p1])

        hist, bins = np.histogram(transformed_img.flatten(), no_of_pixel_values + 1, [0, no_of_pixel_values])
        return hist
