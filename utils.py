import argparse


def parse_float_list(arg):
    """ Parse a bracketed, comma-separated string into a list of floats. """
    if arg.startswith('[') and arg.endswith(']'):
        arg = arg[1:-1]  # Remove the brackets
    try:
        return [float(x) for x in arg.split(',')]
    except ValueError:
        raise argparse.ArgumentTypeError("List of floats must be a bracketed, comma-separated list of floats.")
