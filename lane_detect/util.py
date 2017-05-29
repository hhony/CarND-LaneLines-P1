from numpy import array, ndarray, uint8, \
    zeros, zeros_like
from cv2 import Canny, GaussianBlur, HoughLinesP, \
    cvtColor, fillPoly, line, \
    bitwise_and, addWeighted


def grayscale(image: ndarray, color_order: int) -> ndarray:
    '''
    Applies the Grayscale transform
    :param image: numpy.ndarray input image, video/x-raw dimension
    :param color_order: int     cv2.COLOR_RGB2GRAY,
                                cv2.COLOR_BGR2GRAY with cv2.imread()
    :return: numpy.ndarray
    '''
    return cvtColor(image, color_order)


def canny(image: ndarray, low_threshold: int, high_threshold: int) -> ndarray:
    '''
    Applies the Canny transform
    :param image: numpy.ndarray
    :param low_threshold:   lower bound
    :param high_threshold:  upper bound
    :return: numpy.ndarray
    '''
    return Canny(image, low_threshold, high_threshold)


def gaussian_blur(image: ndarray, kernel=(5,5)) -> ndarray:
    '''
    Applies a Gaussian Noise kernel
    :param image: numpy.ndarray input image
    :param kernel: tuple kernel size
    :return: numpy.ndarray
    '''
    return GaussianBlur(image, kernel, 0)


def get_roi_mask(image: ndarray, vertices=None) -> ndarray:
    '''
    Masks a region of interest
    :param image: numpy.ndarry input image
    :param vertices: numpy.ndarry of x,y tuples
    :return: numpy.ndarray
    '''
    #defining a blank mask to start with
    if vertices is None:
        x_offset = 10
        y_offset = 10
        [y_height, x_width, _] = image.shape
        vertices = array([
            (0, y_height - 1),
            (int(x_width/2 - x_offset), int(y_height/2 + y_offset)),
            (int(x_width/2 + x_offset), int(y_height/2 + y_offset)),
            (x_width - 1, y_height - 1)
        ], dtype=np.int32)
    mask = zeros_like(image)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(image.shape) > 2:
        [_, _, channels] = image.shape # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channels
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    fillPoly(mask, vertices, ignore_mask_color)

    # image only where mask pixels are nonzero
    masked_image = bitwise_and(image, mask)
    return masked_image


def draw_lines(image: ndarray, lines: ndarray, color=None, thickness=2):
    '''
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_image() function below
    """
    :param image: image to apply lines
    :param lines: numpy.ndarray of tuple (x1,y1,x2,y2)
    :param color: tuple (r, g, b) uint8
    :param thickness: pixel width of highlight
    '''
    if color is None:
        color = [255, 0, 0]

    for _line in lines:
        for x1,y1,x2,y2 in _line:
            line(image, (x1, y1), (x2, y2), color, thickness)


def hough_lines(image: ndarray,
                rho: float, theta: float, threshold: int, min_line_len: float, max_line_gap: float,
                with_lines=True) -> ndarray:
    '''
    Hough transformed image
    :param image: numpy.ndarray input image
    :param rho:             distance resolution in pixels of the Hough grid
    :param theta:           angular resolution in radians of the Hough grid
    :param threshold:       minimum number of votes (intersections in Hough grid cell)
    :param min_line_len:    minimum number of pixels to compose line
    :param max_line_gap:    maximum gap in pixels between line segments
    :param with_lines:      if draw line segments on output if true
    :return: numpy.ndarray
    '''
    y_height, x_width, channels = image.shape
    hough_tf = HoughLinesP(image, rho, theta, threshold, array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    if with_lines:
        line_image = zeros((y_height, x_width, channels), dtype=uint8)
        draw_lines(line_image, hough_tf)
        return line_image
    return hough_tf


def weighted_image(image_src: ndarray, image_tf: ndarray, α=0.8, β=1., λ=0.) -> ndarray:
    '''
    Matrix Sum, α(image_src) + β(image_tf) + λ
    :param image_src: preprocessed image
    :param image_tf: hough tf output image
    :param α: scalar image_src
    :param β: scalar image_tf
    :param λ: offset translation
    :return: numpy.ndarray
    '''
    assert image_tf.shape == image_src.shape, 'must be same shape!'
    return addWeighted(image_src, α, image_tf, β, λ)