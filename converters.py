from math import ceil, sqrt
import random

import numpy as np


def rgb_to_xyz(rgb):
    # Observer = 2nd degree, Illuminant = D65
    def helper(octal):
        octal /= 255

        if octal > 0.04045:
            numer = octal + 0.055
            denom = 1.055
            quotient = numer / denom
            octal = quotient ** 2.4
        else:
            octal /= 12.92

        octal *= 100

        return octal

    var_r, var_g, var_b = map(helper, rgb)

    x = (var_r * 0.4124) + (var_g * 0.3576) + (var_b * 0.1805)
    y = (var_r * 0.2126) + (var_g * 0.7152) + (var_b * 0.0722)
    z = (var_r * 0.0193) + (var_g * 0.1192) + (var_b * 0.9505)

    return (x, y, z)


def xyz_to_cielab(xyz, reference=None):
    # Observer = 2nd degree, Illuminant = D65
    if reference is None:
        reference = (95.047, 100.000, 108.883)

    def helper((octal, reference)):
        octal /= reference

        if octal > 0.0085856:
            octal **= (1 / 3)
        else:
            octal = (octal * 7.787) + (16 / 116)

        return octal

    var_x, var_y, var_z = map(helper, zip(xyz, reference))

    l = (116 * var_y) - 16
    a = 500 * (var_x - var_y)
    b = 200 * (var_y - var_z)

    return (l, a, b)


def rgb_to_cielab(rgb):
    return xyz_to_cielab(rgb_to_xyz(rgb))


def cielab_distance_from_rgb(rgb1, rgb2):
    lab1 = rgb_to_cielab(rgb1)
    lab2 = rgb_to_cielab(rgb2)
    return delta_e1994(lab1, lab2)


def delta_e1994(lab1, lab2, weights=None):
    if weights is None:
        weights = (1.0, 1.0, 1.0)

    l1, a1, b1 = lab1
    l2, a2, b2 = lab2

    xC1 = sqrt( (a1 ** 2) + (b1 ** 2) )
    xC2 = sqrt( (a2 ** 2) + (b2 ** 2) )

    xDL = l2 - l1
    xDC = xC2 - xC1
    xDE = distance(lab1, lab2)

    if sqrt(xDE) > ( sqrt(abs(xDL)) + sqrt(abs(xDC)) ):
        xDH = sqrt( (xDE ** 2) - (xDL ** 2) - (xDC ** 2) )
    else:
        xDH = 0

    xSC = 1 + (0.045 * xC1)
    xSH = 1 + (0.015 * xC1)

    xDL /= weights[0]
    xDC /= weights[1] * xSC
    xDH /= weights[2] * xSH

    return sqrt( (xDL ** 2) + (xDC ** 2) + (xDH ** 2) )


def distance(coord1, coord2):
    assert len(coord1) == len(coord2)
    arr1 = np.array(coord1)
    arr2 = np.array(coord2)
    return np.linalg.norm(arr1 - arr2)


def uint_bits(i, num_bytes=1):
    return '{i:0{bits}b}'.format(i=i, bits=num_bytes * 8)


def uint_from_bits(s):
    return int(s, 2)


def uint_tuple_from_bits(s, bytes_per):
    bits_per = bytes_per * 8
    assert len(s) % bits_per == 0

    result = []
    for index in range(0, len(s), bits_per):
        bits = s[index:index + bits_per]
        result.append(uint_from_bits(bits))
    return tuple(result)


def crossover_bits(s1, s2, probability=0.8):
    assert len(s1) == len(s2)

    index = int(ceil(len(s1) * random.uniform(0.0, probability)))
    return s1[:index] + s2[index:]


def crossover_uint_iterables(it1, it2, bytes_per, probability=0.8):
    s1 = "".join([uint_bits(i, num_bytes=bytes_per) for i in it1])
    s2 = "".join([uint_bits(i, num_bytes=bytes_per) for i in it2])
    s3 = crossover_bits(s1, s2, probability)
    return uint_tuple_from_bits(s3, bytes_per)
