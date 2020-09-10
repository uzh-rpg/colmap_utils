#!/usr/bin/env python3

import os
import argparse
from colorama import Fore, init

import utils.exp_utils as eu

init(autoreset=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Generate image list from a folder with respect to another")
    parser.add_argument('--base_dir', required=True, help='starting directory of the reltive path')
    parser.add_argument('--img_dir', required=True, help='directory containing images')
    parser.add_argument('--output_dir', default=None, help='where to write the image list')
    parser.add_argument('--img_nm_to_cam_list', default=None,
                        help='will also make a copy with the new relative paths')

    args = parser.parse_args()

    if not args.output_dir:
        args.output_dir = args.img_dir
    assert os.path.exists(args.base_dir)
    assert os.path.exists(args.img_dir)
    assert os.path.exists(args.output_dir)

    print(Fore.YELLOW + "Going to generate for images in {} w.r.t. {} and save in {}.".format(
        args.img_dir, args.base_dir, args.output_dir))

    all_img_nms = sorted([v for v in os.listdir(args.img_dir)
                          if os.path.splitext(v)[-1] in eu.img_ext])

    rel_path = os.path.relpath(args.img_dir, args.base_dir)

    out_img_list = os.path.join(args.output_dir, 'rel_img_path.txt')
    with open(out_img_list, 'w') as f:
        for v in all_img_nms:
            f.write('{}\n'.format(os.path.join(rel_path, v)))
    print(Fore.GREEN + "Written {} images.".format(len(all_img_nms)))

    if args.img_nm_to_cam_list:
        nm_to_cam_str = eu.readStringsDict(args.img_nm_to_cam_list)
        out_cam_list = os.path.join(args.output_dir, 'rel_img_nm_to_cam_list.txt')
        with open(out_cam_list, 'w') as f:
            for v in all_img_nms:
                if v in nm_to_cam_str:
                    f.write('{} {}\n'.format(os.path.join(rel_path, v), nm_to_cam_str[v]))
                else:
                    print(Fore.RED + "Cannot find camera for {}".format(v))
        print(Fore.GREEN + "Written {} cameras.".format(len(all_img_nms)))
