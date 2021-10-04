#!/usr/bin/env python

import os
import argparse
import re
import cv2


def parse_input():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True,
                        help="directory path with input images")
    parser.add_argument("-w", "--work", required=True,
                        help="directory path for work images")
    parser.add_argument("-o", "--output", required=True,
                        help="directory path for output images and data")
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
    file_root, file_ext = os.path.splitext(os.path.basename(img_file))
    match_id = re.search("[0-9]+", file_root)
    file_id = match_id.group(0) if match_id else ""
    return (file_root, file_ext, file_id)


def write_image_to_work(args, img_file, suffix, img):
    if args["work_save"]:
        dir_work = args['work']
        file_root, file_ext, _ = parse_image_file(img_file)
        cv2.imwrite(f"{dir_work}/{file_root}_{suffix}{file_ext}", img)
    return
