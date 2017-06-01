from numpy import square, sqrt, array, ndarray
from lane_detect.log import logger


# filter thresholds
SLOPE_THRESHOLD  = 0.4
SLOPE_VARIANCE   = 0.025
MAGNITUDE_THRESH = 100
# drawing thresholds
LOWER_X_OFFSET   = 3
UPPER_X_OFFSET   = 5


def sort_slopes(lines: dict) -> list:
    '''
    Sorts slopes into negative and positive sets
    :param lines: dict lines
    :return: <list> [negative, positive]
    '''
    neg_slopes = []; pos_slopes = []
    try:
        for _line in lines:
            [_, _slope, _, _] = lines[_line]
            if _slope < 0:  # right-side
                neg_slopes.append(_slope)
            else:  # left-side
                pos_slopes.append(_slope)
    except Exception as err:
        logger.error('bad sort: %s', err)
    return [neg_slopes, pos_slopes]


def get_slope_stats(slopes: list) -> list:
    '''
    Gives statistics on slope variances
    :param slopes: list of all slopes
    :return: <list> [lane_label, min, max, mean, std]
    '''
    _ret = []
    try:
        for _slopes in slopes:
            _min = 0.0; _max = 0.0
            _lane = '?'; _stdev = []
            if _slopes:
                for _slope in _slopes:
                    if not _slope:
                        continue
                    elif _slope < 0:
                        _lane = 'right'
                    else:
                        _lane = 'left'
                    _min = min(_slope, _min)
                    _max = max(_slope, _max)
                assert len(_slopes) != 0, 'no slopes found'
                _mean = sum(_slopes) / len(_slopes)
                for _slope in _slopes:
                    _stdev.append(square(_slope - _mean))
                assert len(_slopes) != 0, 'no slopes found'
                _std = sum(_stdev) / len(_slopes)
                _ret.append((_lane, _min, _max, _mean, _std))
            else:
                logger.debug('no slopes in input')
    except Exception as err:
        logger.error('bad stats: %s in %s', err, slopes)
    return _ret


def find_dominate_signals(lines: ndarray,
                          slope_thresh=SLOPE_THRESHOLD, magnitude_thresh=MAGNITUDE_THRESH) -> (float, dict):
    '''
    Filters subset of dominate signals in line segments and returns mean slope
    :param lines: <numpy.ndarray> line segments
    :param slope_thresh: filters by slope general lane pitch
    :param magnitude_thresh: filters lines by dominant signal length
    :return: <tuple> (mean_slope: int, signals: dict)
    '''
    _signals = {}; i = 0
    mean_slope = 0.0; max_signal = 0
    try:
        for _line in lines:
            for x1, y1, x2, y2 in _line:
                slope = (y2 - y1) / (x2 - x1)
                magnitude = sqrt((square(x2 - x1) + square(y2 - y1)))
                if magnitude > magnitude_thresh:
                    if slope > slope_thresh or slope < -slope_thresh:
                        logger.debug('points: %s\t slope: %6.3f\t magnitude: %6.3f', (x1, y1, x2, y2), slope, magnitude)
                        if magnitude > max_signal:
                            max_signal = magnitude
                        _signals[i] = [slope, magnitude, (x1, y1), (x2, y2)]
                else:
                    _signals[i] = [slope, magnitude, (x1, y1), (x2, y2)]
                i = i + 1
        _slopes = sort_slopes(_signals)
        slope_stats = get_slope_stats(_slopes)
        avg_slope = []
        for _, _, _, _mean, _ in slope_stats:
            avg_slope.append(abs(_mean))
        assert len(avg_slope) != 0, 'no slopes found'
        mean_slope = float(sum(avg_slope) / len(avg_slope))
    except Exception as err:
        logger.error('bad signal: %s in %s', err, _signals)
    return mean_slope, _signals


def interpolate_dominate_lines(region_mask: array, signals: dict, mean_slope: float, lower_bound: int, upper_bound: int,
                               slope_variance=SLOPE_VARIANCE) -> dict:
    '''
    Interpolates lines based on ROI mask and mean_slope
    :param region_mask: ROI mask shape
    :param signals: dominate signals in image
    :param mean_slope: mean of dominate signals
    :param lower_bound: lower y value in image
    :param upper_bound: upper y value in image
    :param slope_variance: acceptable slope variance
    :return: <dict> interpolated lines
    '''
    _slots = {}
    try:
        for _line in signals:
            [_slope, _, (x1, y1), (x2, y2)] = signals[_line]
            if (abs(_slope) >= mean_slope - slope_variance) or (abs(_slope) <= mean_slope + slope_variance):
                if region_mask[y1][x1][0] and region_mask[y2][x2][0]:
                    logger.debug('%s: %s\t %s: %s', (x1, y1), region_mask[y1][x1][0], (x2, y2), region_mask[y2][x2][0])
                    _offset = float(y1 / _slope - x1)
                    new_p1 = (int(lower_bound / _slope - _offset), lower_bound)
                    new_p2 = (int(upper_bound / _slope - _offset), upper_bound)
                    _slots[_line] = [_line, _slope, new_p1, new_p2]
    except Exception as err:
        logger.error('interpolation error: %s', err)
    return _slots


def get_point_stats(edges: list, lower_bound: int) -> (int, int):
    '''
    Averages x values and returns mean tuple (x1, x2)
    :param edges: dict on line info
    :param lower_bound: lower y value
    :return: <tuple> (mean_x1: int, mean_x2: int)
    '''
    min_x1 = 0; min_x2 = 0
    max_x1 = 0; max_x2 = 0
    cnt_x1 = []; cnt_x2 = []
    for _line in edges:
        _, _, (x1,y1), (x2,y2) = edges[_line]
        if lower_bound == y1:
            min_x1 = min(x1, min_x1)
            max_x1 = max(x1, max_x1)
            min_x2 = min(x2, min_x2)
            max_x2 = max(x2, max_x2)
            cnt_x1.append(x1)
            cnt_x2.append(x2)
        else:
            min_x2 = min(x1, min_x1)
            max_x2 = max(x1, max_x1)
            min_x1 = min(x2, min_x2)
            max_x1 = max(x2, max_x2)
            cnt_x2.append(x1)
            cnt_x1.append(x2)
    mean_x1 = int(sum(cnt_x1) / len(cnt_x1))
    mean_x2 = int(sum(cnt_x2) / len(cnt_x2))
    return mean_x1, mean_x2


def convert_lane_edges_to_polygons(edges: dict, lower_bound: int, upper_bound: int,
                                   lower_x_offset=LOWER_X_OFFSET, upper_x_offset=UPPER_X_OFFSET) -> (array, array):
    '''
    Converts interpolated lines lanes into lane polygons
    :param edges: dict of line info
    :param lower_bound: lower y value in image
    :param upper_bound: upper y value in image
    :return: <tuple> (left_lane: array, right_lane: array)
    '''
    left = {}; right = {}
    for _line in edges:
        [_, _slope, _, _] = edges[_line]
        if _slope < 0:
            right[_line] = edges[_line]
        else:
            left[_line] = edges[_line]

    mean_left_lower, mean_left_upper = get_point_stats(left, lower_bound)
    poly_left = array([
        ((mean_left_lower - lower_x_offset), lower_bound),
        ((mean_left_upper - upper_x_offset), upper_bound),
        ((mean_left_upper + upper_x_offset), upper_bound),
        ((mean_left_lower + lower_x_offset), lower_bound)
    ])
    mean_right_lower, mean_right_upper = get_point_stats(right, lower_bound)
    poly_right = array([
        ((mean_right_lower - lower_x_offset), lower_bound),
        ((mean_right_upper - upper_x_offset), upper_bound),
        ((mean_right_upper + upper_x_offset), upper_bound),
        ((mean_right_lower + lower_x_offset), lower_bound)
    ])
    logger.debug('polygons (left-lane, right-lane):\n(array(%s),\narray(%s))', poly_left, poly_right)
    return poly_left, poly_right