from numpy import square, sqrt, array, ndarray
from lane_detect.log import logger


SLOPE_THRESHOLD = 0.4
SLOPE_VARIANCE = 0.025
MAGNITUDE_THRESH = 100


def find_dominate_signals(lines: ndarray):
    _signals = {}; i = 0
    max_slope = 0; max_signal = 0
    try:
        for _line in lines:
            for x1, y1, x2, y2 in _line:
                slope = (y2 - y1) / (x2 - x1)
                magnitude = sqrt((square(x2 - x1) + square(y2 - y1)))
                if magnitude > MAGNITUDE_THRESH:
                    if slope > SLOPE_THRESHOLD or slope < -SLOPE_THRESHOLD:
                        logger.debug('points: %s\t slope: %6.3f\t magnitude: %6.3f', (x1, y1, x2, y2), slope, magnitude)
                        if magnitude > max_signal:
                            max_slope = abs(slope)
                            max_signal = magnitude
                        _signals[i] = [slope, magnitude, (x1, y1), (x2, y2)]
                else:
                    _signals[i] = [slope, magnitude, (x1, y1), (x2, y2)]
                i = i + 1
    except Exception as err:
        logger.error('bad signal: %s', err)
    return max_slope, _signals


def interpolate_dominate_lines(region_mask: array, signals: dict, max_slope: float, lower_bound: int, upper_bound: int,
                               slope_variance=SLOPE_VARIANCE) -> dict:
    _slots = {}
    try:
        for _line in signals:
            [_slope, _, (x1, y1), (x2, y2)] = signals[_line]
            if (abs(_slope) >= max_slope - slope_variance) or (abs(_slope) <= max_slope + slope_variance):
                if region_mask[y1][x1][0] and region_mask[y2][x2][0]:
                    logger.debug('%s: %s\t %s: %s', (x1, y1), region_mask[y1][x1][0], (x2, y2), region_mask[y2][x2][0])
                    _offset = float(y1 / _slope - x1)
                    new_p1 = (int(lower_bound / _slope - _offset), lower_bound)
                    new_p2 = (int((upper_bound - 1) / _slope - _offset), (upper_bound - 1))
                    _slots[_line] = [_line, _slope, new_p1, new_p2]
    except Exception as err:
        logger.error('interpolation error: %s', err)
    return _slots


def sort_slopes(slopes: dict) -> list:
    neg_slopes = []; pos_slopes = []
    try:
        for _line in slopes:
            [_, _slope, _, _] = slopes[_line]
            if _slope < 0:  # right-side
                neg_slopes.append(_slope)
            else:  # left-side
                pos_slopes.append(_slope)
    except Exception as err:
        logger.error('bad sort: %s', err)
    return [neg_slopes, pos_slopes]


def get_slope_stats(slopes: list) -> list:
    _ret = []
    try:
        for _slopes in slopes:
            _min = 0.0; _max = 0.0
            _direction = 0; _stdev = []
            for _slope in _slopes:
                if _slope < 0:
                    _direction = 'right'
                else:
                    _direction = 'left'
                _min = min(_slope, _min)
                _max = max(_slope, _max)
            _mean = sum(_slopes) / len(_slopes)
            for _slope in _slopes:
                _stdev.append(square(_slope - _mean))
            _std = sum(_stdev) / len(_slopes)
            _ret.append((_direction, _min, _max, _mean, _std))
    except Exception as err:
        logger.error('bad stats: %s', err)
    return _ret
