import numpy as np
from math import factorial as fac
import scipy.signal as sig
import scipy.ndimage as ndi
import imageio
import skimage.color
import matplotlib.pyplot as plt
import skimage.data
import os

NORMALIZE = 255
DIM_MIN = 16


def binomial(x, y):
    binom = fac(x) // fac(y) // fac(x - y)
    return binom

def pascal(m):
    return [binomial(m - 1, y) for y in range(m)]

def create_filter_vec(binomes):
    filter_vec = list()
    k = len(binomes)
    numerator = 1/2**(k-1)
    for bin in binomes:
        filter_vec.append(bin*numerator)
    return np.array(filter_vec)

def read_image(filename, representation):
    """
    Function that reads an image file and convert it into a given representation
    :param filename: the filename of an image on disk
    :param representation: representation code, either 1 or 2 defining whether the output should
                           be a grayscale image (1) or an RGB image (2)
    :return: an image represented by a matrix of type np.float64
    """

    color_flag = True #if RGB image
    image = imageio.imread(filename)

    float_image = image.astype(np.float64)

    if not np.all(image <= 1):
        float_image /= NORMALIZE #Normalized to range [0,1]

    if len(float_image.shape) != 3 : #Checks if RGB or Grayscale
        color_flag = False

    if color_flag and representation == 1 : #Checks if need RGB to Gray
        return skimage.color.rgb2gray(float_image)

    # Same coloring already
    return float_image


def build_gaussian_pyramid(im, max_levels, filter_size):
    """
    This function build a Gaussian pyramid
    :param im: a grayscale image with double values in [0,1]
    :param max_levels: the maximum number of levels in the resulting pyramid
    :param filter_size: the size of the gaussian filter (an odd scalar that represents a squared filter) to
                        be used in constructing the pyramid filter
    :return: pyr, filter_vec
    """
    binom_vec = create_filter_vec(pascal(filter_size))
    filter_vec = binom_vec.reshape(1, filter_size)
    filter = sig.convolve2d(filter_vec, filter_vec.T)

    pyramid = list()
    pyramid.append(im)

    for index in range(max_levels - 1):
        blured = ndi.filters.convolve(im, filter)
        blured = blured[::2,::2]


        height, width = blured.shape
        if height <= DIM_MIN or width <= DIM_MIN:
            break

        pyramid.append(blured)
        # plt.imshow(blured, cmap='gray')  # TODO a enlever
        # plt.show()
        im = blured

    return pyramid, filter_vec

def expand(im, filter_vec):
    height, width = im.shape
    expanded = np.zeros((height*2, width*2)) # pad with zeros
    expanded[::2,::2] = im # Getting the values inside
    blured_filter = 4 * sig.convolve2d(filter_vec, filter_vec.T)
    return ndi.filters.convolve(expanded, blured_filter)



def build_laplacian_pyramid(im, max_levels, filter_size):
    """
    This function build a Laplacian pyramid
    :param im: a grayscale image with double values in [0,1]
    :param max_levels: the maximum number of levels in the resulting pyramid
    :param filter_size: the size of the gaussian filter (an odd scalar that represents a squared filter) to
                        be used in constructing the pyramid filter
    :return: pyr, filter_vec
    """

    pyramid = list()

    gauss_pyramid, filter_vec = build_gaussian_pyramid(im, max_levels, filter_size)

    for index in range(len(gauss_pyramid) - 1):
        expand_gauss = expand(gauss_pyramid[index + 1], filter_vec)
        new_im = gauss_pyramid[index] - expand_gauss
        pyramid.append(new_im)
        # plt.imshow(new_im, cmap='gray') #TODO a enlever
        # plt.show()

    pyramid.append(gauss_pyramid[-1])
    # plt.imshow(gauss_pyramid[-1], cmap='gray')  # TODO a enlever
    # plt.show()

    return pyramid, filter_vec

def laplacian_to_image(lpyr, filter_vec, coeff):
    """
    This function implements the reconstruction of an image from its Laplacian pyramid
    :param lpyr: Laplacian pyramid
    :param filter_vec: filter generated
    :param coeff: python list - length is the same as the number of levels in the lpyr
    :return: img
    """
    img = lpyr[-1]

    for index in range(len(lpyr)-2, -1, -1):
        expanded = expand(img, filter_vec)
        img = (lpyr[index] * coeff[index]) + expanded
    return img

def render_pyramid(pyr, levels):
    """
    This function renders the given pyramid into an image
    :param pyr: Gaussian or Laplacian pyramid
    :param level: number of levels to present in the result <= max_levels
    :return: single black image in which the pyramid levels of the given pyramid pyr are stacked horizontally
    """
    img = (pyr[0] - pyr[0].min())/(pyr[0].max() - pyr[0].min())
    heigh_use = 0

    for index in range(1, levels):
        heigh, width = pyr[index].shape
        stretch_im = (pyr[index] - pyr[index].min())/(pyr[index].max() - pyr[index].min())
        heigh_use += heigh
        zeros_arr = np.zeros(shape=(heigh_use, width))
        img_add = np.concatenate((stretch_im, zeros_arr))
        img = np.concatenate((img, img_add), axis=1)

    return img


def display_pyramid(pyr, levels):
    """
    This function displays the stacked pyramid
    :param pyr: Gaussian or Laplacian pyramid
    :param levels: levels to present
    :return: display pyramid
    """
    plt.imshow(render_pyramid(pyr, levels), cmap='gray')
    plt.show()

def pyramid_blending(im1, im2, mask, max_levels, filter_size_im, filter_size_mask):
    """
    this function implements a pyramid blending
    :param im1: input grayscale image to be blended
    :param im2: input grayscale image to be blended
    :param mask: boolean containing which parts of im1 and im2 should appear in the resulting im_blend
    :param max_levels: parameter to use when generating the Gaussian or Laplacian pyramids
    :param filter_size_im: size of the gaussian filter defining the filter used in the construction of the
                           im1 and im2 pyramids
    :param filter_size_mask: size of the gaussian filter which defining the filter used in the construction
                             of the gaussian pyramid of mask
    :return: im_blend - blended image
    """
    # Step 1
    L1, filter1 = build_laplacian_pyramid(im1, max_levels, filter_size_im)
    L2, filter2 = build_laplacian_pyramid(im2, max_levels, filter_size_im)

    # Step 2
    Gm, filter_m = build_gaussian_pyramid(mask.astype(np.float64), max_levels,  filter_size_mask)

    # Step 3
    pyr = list()
    for index in range(len(Gm)):
        pyr.append((Gm[index]*L1[index]) + ((1 - Gm[index]) * (L2[index])))

    # Step 4
    coeffs = np.ones(len(Gm))
    Lout = laplacian_to_image(pyr, filter1, coeffs)

    return Lout.clip(0, 1)

def relpath(path):
    return os.path.join(os.path.dirname(__file__), path)

def blending_example1():
    jap = relpath("externals/templejapon.jpg")
    plage = relpath("externals/plageee.jpg")
    mask = relpath("externals/mask1.jpg")

    toilets_im = read_image(jap, 2)
    squat_im = read_image(plage, 2)
    mask_im = read_image(mask, 1)

    mask_im[mask_im < 0.5] = 0
    mask_im[mask_im >= 0.5] = 1
    mask_im = mask_im.astype(np.bool)
    r = pyramid_blending(squat_im[:,:,0], toilets_im[:,:,0], mask_im, 4, 3, 3)
    g = pyramid_blending(squat_im[:,:,1], toilets_im[:,:,1], mask_im, 4, 3, 3)
    b = pyramid_blending(squat_im[:,:,2], toilets_im[:,:,2], mask_im, 4, 3, 3)

    blended = np.zeros((toilets_im.shape[0], toilets_im.shape[1], toilets_im.shape[2]))
    blended[:,:,0] = r
    blended[:,:,1] = g
    blended[:,:,2] = b

    plt.figure()
    plt.subplot(2,2,1)
    plt.axis('off')
    plt.imshow(squat_im)
    plt.subplot(2,2,2)
    plt.axis('off')
    plt.imshow(toilets_im)
    plt.subplot(2,2,3)
    plt.axis('off')
    plt.imshow(mask_im, cmap='gray')
    plt.subplot(2,2,4)
    plt.axis('off')
    plt.imshow(blended)
    plt.show()

    # plt.axis('off')
    # plt.imshow(blended)
    # plt.show()


    return squat_im, toilets_im, mask_im, blended.astype(np.float64)

def blending_example2():
    lune = relpath("externals/lune.jpg")
    pyra = relpath("externals/pyra.jpg")
    mask = relpath("externals/mask2.jpg")

    toilets_im = read_image(lune, 2)
    squat_im = read_image(pyra, 2)
    mask_im = read_image(mask, 1)

    mask_im[mask_im < 0.5] = 0
    mask_im[mask_im >= 0.5] = 1
    mask_im = mask_im.astype(np.bool)
    r = pyramid_blending(squat_im[:,:,0], toilets_im[:,:,0], mask_im, 4, 3, 3)
    g = pyramid_blending(squat_im[:,:,1], toilets_im[:,:,1], mask_im, 4, 3, 3)
    b = pyramid_blending(squat_im[:,:,2], toilets_im[:,:,2], mask_im, 4, 3, 3)

    blended = np.zeros((toilets_im.shape[0], toilets_im.shape[1], toilets_im.shape[2]))
    blended[:,:,0] = r
    blended[:,:,1] = g
    blended[:,:,2] = b

    plt.figure()
    plt.subplot(2,2,1)
    plt.axis('off')
    plt.imshow(squat_im)
    plt.subplot(2,2,2)
    plt.axis('off')
    plt.imshow(toilets_im)
    plt.subplot(2,2,3)
    plt.axis('off')
    plt.imshow(mask_im, cmap='gray')
    plt.subplot(2,2,4)
    plt.axis('off')
    plt.imshow(blended)
    plt.show()

    # plt.axis('off')
    # plt.imshow(blended)
    # plt.show()


    return squat_im, toilets_im, mask_im, blended.astype(np.float64)


# if __name__ == "__main__":
#     # print(pascal(5))
#     im = skimage.color.rgb2gray(skimage.data.astronaut())
#     # plt.imshow(im, cmap='gray')
#     # plt.show()
#     # pyr, vec = build_gaussian_pyramid(im, 5, 3)
#     # plt.imshow(blured, cmap='gray')
#     # plt.show()
#     pyr, vec = build_laplacian_pyramid(im, 5, 3)
#
#     # img = laplacian_to_image(pyr, vec, [1,1,1,1,1])
#     # plt.imshow(img, cmap='gray')
#     # plt.show()
#
#     # display_pyramid(pyr, 5)
#     blending_example1()
#     blending_example2()