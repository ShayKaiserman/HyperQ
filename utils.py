import argparse
from scipy.signal import savgol_filter

PRINT_START = '\033['
RED = '91m'
BLUE = '94m'
GREEN = '32m'
PRINT_STOP = '\033[0m'


def parse_float_list(arg):
    """ Parse a bracketed, comma-separated string into a list of floats. """
    if arg.startswith('[') and arg.endswith(']'):
        arg = arg[1:-1]  # Remove the brackets
    try:
        return [float(x) for x in arg.split(',')]
    except ValueError:
        raise argparse.ArgumentTypeError("List of floats must be a bracketed, comma-separated list of floats.")


def adaptive_savgol_filter(data, min_window_length=5, max_window_length=1001, poly_order=3):
    length = len(data)
    if length < min_window_length:
        return data  # Return original data if we have too few points

    # Use an odd window length, not exceeding the data length or max_window_length
    window_length = min(length if length % 2 != 0 else length - 1, max_window_length)
    window_length = max(window_length, min_window_length)

    return savgol_filter(data, window_length, poly_order)