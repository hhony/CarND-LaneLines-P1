from numpy import square, sqrt, array, ndarray
from lane_detect.log import logger


# filter thresholds
SLOPE_THRESHOLD  = 0.08
SLOPE_MAX_CUTOFF = 0.9
SLOPE_VARIANCE   = 0.25
MAGNITUDE_THRESH = 100
# drawing thresholds
LOWER_X_OFFSET   = 5
UPPER_X_OFFSET   = 10


def sort_slopes(lines: dict, slope_thresh=SLOPE_THRESHOLD) -> list:
    '''
    Sorts slopes into negative and positive sets
    :param lines: dict lines
    :return: <list> [negative, positive]
    '''
    neg_slopes = []; pos_slopes = []
    try:
        for _line in lines:
            _slope = lines[_line]['slope']
            abs_slope = abs(_slope)
            if  abs_slope > slope_thresh:
                if _slope < 0: # likely right-side
                    neg_slopes.append(_slope)
                else: # likely left-side
                    pos_slopes.append(_slope)
            else:
                logger.warning('throwing out %s, mag: %s', lines[_line]['slope'], lines[_line]['magnitude'])
    except Exception as err:
        logger.error('bad sort: %s', err)
    return [neg_slopes, pos_slopes]


def get_slope_stats(slopes: list, threshold=SLOPE_THRESHOLD) -> dict:
    '''
    Gives statistics on slope variances
    :param slopes: list of all slopes
    :return: <list> [lane_label, min, max, mean, std]
    '''
    _ret = {}
    try:
        for _slopes in slopes:
            _min = -threshold; _max = threshold
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
                _std = sum(_stdev) / len(_slopes)
                _ret[_lane] = {
                    'min': _min,
                    'max': _max,
                    'mean': _mean,
                    'std': _std
                }
            else:
                logger.debug('no slopes in input')
    except Exception as err:
        logger.error('bad stats: %s in %s', err, slopes)
    return _ret


def valid_within_fov(points: list, region_mask: array) -> bool:
    '''
    Validates if point are withing the region mask
    :param points: list tuple (x: int, y: int)
    :param region_mask: array([bool])
    :return: True if valid
    '''
    for _x, _y in points:
        if region_mask[_y][_x][0]:
            return True
    return False


def find_dominate_signals(lines: ndarray, signals: dict, region_mask: array,
                          slope_max_cutoff=SLOPE_MAX_CUTOFF, slope_thresh=SLOPE_THRESHOLD,
                          magnitude_thresh=MAGNITUDE_THRESH) -> float:
    '''
    Filters subset of dominate signals in line segments and returns mean slope
    :param lines: <numpy.ndarray> line segments
    :param signals dict filtered by one point valid in ROI
    :param region_mask: ROI mask shape
    :param slope_thresh: filters by slope general lane pitch
    :param magnitude_thresh: filters lines by dominant signal length
    :return: mean_slope: int of dominant signals
    '''
    i = 0; max_slope = 0.0; min_slope = 0.0; max_signal = 0; x_valid = 0
    try:
        for _line in lines:
            for x1, y1, x2, y2 in _line:
                if x1 == x2 or y1 == y2:
                    logger.warning('disregarding: %s, %s', (x1, y1), (x2, y2))
                    continue
                if not valid_within_fov([(x1, y1), (x2, y2)], region_mask):
                    logger.debug('invalid in FOV, thowing out %s', (x1,y1,x2,y2))
                    continue
                slope = (y2 - y1) / (x2 - x1)
                assert slope != 0, 'slope is zero {0}'.format(_line)
                offset = (y1 / slope) - x1
                magnitude = sqrt((square(x2 - x1) + square(y2 - y1)))
                abs_slope = abs(slope)
                if magnitude > magnitude_thresh and abs_slope > slope_thresh and abs_slope < slope_max_cutoff:
                        if min_slope == 0:
                            min_slope = abs_slope
                        else:
                            min_slope = min(abs_slope, min_slope)
                        logger.debug('points: %s\t slope: %6.3f\t magnitude: %6.3f', (x1, y1, x2, y2), slope, magnitude)
                        if magnitude > max_signal:
                            max_slope = abs(slope)
                            max_signal = magnitude
                signals[i] = {
                    'slope': slope,
                    'offset': offset,
                    'magnitude': magnitude,
                    'p1': (x1, y1),
                    'p2': (x2, y2)
                }
                i = i + 1
    except Exception as err:
        logger.error('bad signal: %s in %s', err, signals)
    mean_slope = (max_slope + min_slope) / 2
    logger.debug('mean_slope: %s, max: %s, min: %s, max_signal: %s', mean_slope, max_slope, min_slope, max_signal)
    return mean_slope


def find_mean_slope(signals: dict, slope_thresh=SLOPE_THRESHOLD) -> float:
    '''
    Finds the mean slope from provided line data
    :param signals: dict line data
    :param slope_thresh: float default mean, which could be dominant mean..
    :return: float new mean
    '''
    try:
        _slopes = sort_slopes(signals)
        slope_stats = get_slope_stats(_slopes)
        cnt = 0; avg_slope = 0.0
        if 'mean' in slope_stats:
            for _side in slope_stats:
                avg_slope = avg_slope + abs(slope_stats[_side]['mean'])
                cnt = cnt + 1
            assert cnt != 0, 'no elements'
            slope_thresh = float(avg_slope / cnt)
    except Exception as err:
        logger.error('bad mean: %s', err)
    return slope_thresh


def interpolate_dominate_lines(signals: dict, interpolations: dict,
                               mean_slope: float, lower_bound: int, upper_bound: int, horizontal_limit: int,
                               slope_variance=SLOPE_VARIANCE):
    '''
    Interpolates lines based on ROI mask and mean_slope
    :param signals: dominate signals in image
    :param mean_slope: mean of dominate signals
    :param lower_bound: int lower y value in image
    :param upper_bound: int upper y value in image
    :param horizontal_limit int maximum possible x-value
    :param slope_variance: acceptable slope variance
    '''
    try:
        for _line in signals:
            _slope  = signals[_line]['slope']
            _offset = signals[_line]['offset']
            x1, y1  = signals[_line]['p1']
            x2, y2  = signals[_line]['p2']
            if not _slope:
                continue
            abs_slope = abs(_slope)
            delta = abs(abs_slope - mean_slope)
            if delta < slope_variance:
                logger.debug('%s\t %s', (x1, y1), (x2, y2))
                assert _slope != 0, 'slope is zero in signals'
                new_p1 = (int(lower_bound / _slope - _offset), lower_bound)
                new_p2 = (int(upper_bound / _slope - _offset), upper_bound)
                is_valid = True
                for _x, _y in [new_p1, new_p2]:
                    if _x < 0:
                        logger.warning('line extends too far left, throwing out %s', (_x, _y))
                        is_valid = False
                    elif _x > horizontal_limit:
                        logger.warning('line extends too far right, throwing out %s', (_x, _y))
                        is_valid = False
                if is_valid:
                    interpolations[_line] = {
                        'slope': _slope,
                        'offset': _offset,
                        'p1': new_p1,
                        'p2': new_p2
                    }
    except Exception as err:
        logger.error('interpolation error: %s', err)


def get_point_stats(edges: dict, lower_bound: int) -> (int, int):
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
        x1, y1 = edges[_line]['p1']
        x2, y2 = edges[_line]['p2']
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
    assert len(cnt_x1) != 0, 'len cnt_x1 is zero'
    assert len(cnt_x2) != 0, 'len cnt_x2 is zero'
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
        _slope = edges[_line]['slope']
        if _slope < 0:
            right[_line] = edges[_line]
        else:
            left[_line] = edges[_line]

    if left:
        mean_left_lower, mean_left_upper = get_point_stats(left, lower_bound)
        poly_left = array([
            ((mean_left_lower - lower_x_offset), lower_bound),
            ((mean_left_upper - upper_x_offset), upper_bound),
            ((mean_left_upper + upper_x_offset), upper_bound),
            ((mean_left_lower + lower_x_offset), lower_bound)
        ])
    else:
        poly_left = array(None)

    if right:
        mean_right_lower, mean_right_upper = get_point_stats(right, lower_bound)
        poly_right = array([
            ((mean_right_lower - lower_x_offset), lower_bound),
            ((mean_right_upper - upper_x_offset), upper_bound),
            ((mean_right_upper + upper_x_offset), upper_bound),
            ((mean_right_lower + lower_x_offset), lower_bound)
        ])
    else:
        poly_right = array(None)

    logger.debug('polygons (left-lane, right-lane):\n(array(%s),\narray(%s))', poly_left, poly_right)
    return poly_left, poly_right