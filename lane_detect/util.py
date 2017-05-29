from os.path import isfile
from numpy import array, ndarray, uint8, int32, pi, \
    zeros, zeros_like
from cv2 import Canny, GaussianBlur, HoughLinesP, \
    imread, cvtColor, COLOR_BGR2GRAY, COLOR_RGB2GRAY, \
    fillPoly, line, \
    bitwise_and, addWeighted
from lane_detect.log import logger


class LaneFilter(object):
    def __init__(self, image=None, filename=None):
        '''
        LaneFilter will perform image transforms on itself
        :param image: <numpy.nd_array>
        :param filename: file path <str>
        '''
        if filename:
            assert issubclass(str, type(filename)), 'image path must be <str>'
            if isfile(filename):
                self.image = imread(filename)
                self.image_tf = zeros_like(self.image)
                self.gray = self.grayscale(image=self.image, color_order=COLOR_BGR2GRAY)
                self.mask = zeros_like(self.image)
                self.lane = zeros_like(self.image)
            else:
                RuntimeError('{0} not found.'.format(filename))
        elif image:
            assert issubclass(ndarray, type(image)), 'image must be <numpy.ndarray>'
            self.image = image
            self.image_tf = zeros_like(self.image)
            self.gray = self.grayscale(image=self.image)
            self.mask = zeros_like(self.image)
            self.lane = zeros_like(self.image)
        else:
            raise RuntimeError('must provide either filename <str> or image <numpy.ndarray>')

    def grayscale(self, image=None, color_order=COLOR_RGB2GRAY) -> ndarray:
        '''
        Applies the Grayscale transform
        :param image: numpy.ndarray input image, video/x-raw dimension
        :param color_order: int     cv2.COLOR_RGB2GRAY,
                                    cv2.COLOR_BGR2GRAY with cv2.imread()
        :return: numpy.ndarray
        '''
        if image is not None:
            assert issubclass(ndarray, type(image)), 'image must be <numpy.ndarray>'
            return cvtColor(image, color_order)
        self.gray = cvtColor(self.image, color_order)
        return self.gray

    def gaussian_blur(self, image=None, kernel=(5,5)) -> ndarray:
        '''
        Applies a Gaussian Noise kernel
        :param image: numpy.ndarray input image
        :param kernel: tuple kernel size
        :return: numpy.ndarray
        '''
        if image is not None:
            assert issubclass(ndarray, type(image)), 'image must be <numpy.ndarray>, for gaussian blur'
            return GaussianBlur(image, kernel, 0)
        self.image_tf = GaussianBlur(self.gray, kernel, 0)
        return self.image_tf

    def canny_edges(self, low_threshold: int, high_threshold: int, image=None) -> ndarray:
        '''
        Applies the Canny transform
        :param image: numpy.ndarray
        :param low_threshold:   lower bound
        :param high_threshold:  upper bound
        :return: numpy.ndarray
        '''
        if image is None:
            self.image_tf = Canny(self.image_tf, low_threshold, high_threshold)
            return  self.image_tf
        else:
            assert issubclass(ndarray, type(image)), 'image must be <numpy.ndarray>, for canny edges'
        return Canny(image, low_threshold, high_threshold)

    def get_roi_mask(self, image=None, vertices=None) -> ndarray:
        '''
        Masks a region of interest
        :param image: numpy.ndarray input image
        :param vertices: numpy.ndarray of x,y tuples
        :return: numpy.ndarray
        '''
        if image is None:
            image = self.image
        else:
            assert issubclass(ndarray, type(image)), 'image must be <numpy.ndarray>, to get roi mask'
        #defining a blank mask to start with
        if vertices is None:
            x_offset = 10
            y_offset = 45
            [y_height, x_width, _] = image.shape
            vertices = array([
                (0, y_height - 1),
                (int(x_width/2 - x_offset), int(y_height/2 + y_offset)),
                (int(x_width/2 + x_offset), int(y_height/2 + y_offset)),
                (x_width - 1, y_height - 1)
            ], dtype=int32)
        self.mask = zeros_like(image)
        # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
        if len(image.shape) > 2:
            [_, _, channels] = image.shape # i.e. 3 or 4 depending on your image
            _mask_color = (255,) * channels
        else:
            _mask_color = 255
        # filling pixels inside the polygon defined by "vertices" with the fill color
        if vertices.dtype.name != 'int32':
            fillPoly(self.mask, int32([vertices]), _mask_color)
        else:
            fillPoly(self.mask, [vertices], _mask_color)
        # image only where mask pixels are nonzero
        self.mask = bitwise_and(image, self.mask)
        return self.mask

    def apply_roi_mask(self, image=None):
        if image is not None:
            assert self.image.shape == image.shape, 'images must be same shape, for roi to work'
            self.image_tf = image
        self.image_tf = bitwise_and(self.image_tf, self.get_roi_mask())

    def draw_lines(self, lines: ndarray, image=None, color=None, thickness=2) -> ndarray:
        '''
        Lines are drawn on the image inplace.
        """
        :param image: image to apply lines
        :param lines: numpy.ndarray of tuple (x1,y1,x2,y2)
        :param color: tuple (r, g, b) uint8
        :param thickness: pixel width of highlight
        '''
        if image is not None:
            assert image.shape == self.image.shape, 'images must be same shape, to draw lines'
            self.image_tf = image
        y_height, x_width, channels = self.image.shape
        self.image_tf = zeros((y_height, x_width, channels), dtype=uint8)
        # assign default color
        if color is None:
            color = [255, 0, 0]
        # draw lines
        logger.debug('-----------------------------------------')
        for _line in lines:
            for x1,y1,x2,y2 in _line:
                m = (y2-y1) / (x2-x1)
                logger.debug('slope: %s: %s', (x1,y1,x2,y2), m)
                line(self.image_tf, (x1, y1), (x2, y2), color, thickness)
        return self.image_tf

    def hough_lines(self, rho: float, threshold: int, min_line_len: float, max_line_gap: float,
                    theta=pi/180, image=None, with_lines=True) -> ndarray:
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
        if image is None:
            image = self.image_tf
        else:
            assert issubclass(ndarray, type(image)), 'image must be <numpy.ndarray>, for hough tf'
        hough_tf = HoughLinesP(image, rho, theta, threshold, array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
        if with_lines:
            return self.draw_lines(lines=hough_tf)
        return hough_tf

    def weighted_image(self, image_tf=None, α=0.8, β=1., λ=0.) -> ndarray:
        '''
        Matrix Sum, α(image_src) + β(image_tf) + λ
        :param image_tf: hough tf output image
        :param α: scalar image_src
        :param β: scalar image_tf
        :param λ: offset translation
        :return: numpy.ndarray
        '''
        if image_tf is not None:
            self.image_tf = image_tf
        assert self.image_tf.shape == self.image.shape, 'must be same shape!'
        self.lane = addWeighted(self.image, α, self.image_tf, β, λ)
        return self.lane
