#!/usr/bin/env python3

import os
import argparse
import re
import cv2


def parse_input():
    """
    Parse command line arguments.

    Returns
    -------
    args : dict of (string, values)
        Dictionary of command line arguments.
        - args["input"] is the directory of input images to detect and classify;
        - args["output"] is the directory for output images and txt files to be
          written to;
        - args["digits"] is the directory of digit and directional arrow
          training images;
        - args["work"] is the directory that work (pipeline) images will be
          written to;
        - args["work_save"] is a flag indicating if work images are to be
          constructed and saved, or not.
    img_files : list of string
        The set of input images on which the detection and classification
        algorithms are to be run.

    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True,
                        help="directory path with input images")
    parser.add_argument("-o", "--output", required=True,
                        help="directory path for output images and data")
    parser.add_argument("-d", "--digits", required=True,
                        help="directory path for digit and arrow images")
    parser.add_argument("-w", "--work", required=True,
                        help="directory path for work images")
    parser.add_argument("-W", "--work-save", action="store_true",
                        help="flag if intermediate images " +
                        "are to be saved to work directory")

    args = vars(parser.parse_args())

    dir_input = args["input"]

    img_files = [os.path.join(dir_input, f)
                 for f in os.listdir(dir_input)
                 if os.path.isfile(os.path.join(dir_input, f))
                 and os.path.splitext(f)[1] in {".jpg", ".png"}]

    return args, img_files


def parse_image_file(img_file):
    """
    Parse an image filepath into rootname, extension, and the integer ID.

    Parameters
    ----------
    img_file : string

    Returns
    -------
    file_root : string
        Root of the basename of `img_file`.
    file_ext : string
        Extension of the basename of `img_file`.
    file_id : string
        Integer ID of `img_file`.

    """
    file_root, file_ext = os.path.splitext(os.path.basename(img_file))
    match_id = re.search("[0-9]+", file_root)
    file_id = match_id.group(0) if match_id else ""
    return (file_root, file_ext, file_id)
