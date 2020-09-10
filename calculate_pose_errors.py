#!/usr/bin/env python3

import os
import argparse
from colorama import Fore, init

import utils.exp_utils as eu
from utils.colmap_read_model import read_images_binary

init(autoreset=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Calculate registeraition error")
    parser.add_argument('--reg_model_dir', required=True, help='model folder containing all images')
    parser.add_argument('--reg_img_name_to_colmap_Tcw', required=True,
                        help='groundtruth poses, colmap format')
    parser.add_argument('--reg_img_dir', required=True, help='used to compute the relative path')
    parser.add_argument('--output_path', required=True, help='where to write the pose errors')
    args = parser.parse_args()

    base_img_dir = os.path.abspath(os.path.join(args.reg_model_dir, '../images'))
    print(Fore.GREEN + "Base image folder is {}".format(base_img_dir))
    rel_path = os.path.relpath(args.reg_img_dir, base_img_dir)

    reg_names, reg_qtvec = eu.readNamedValues(args.reg_img_name_to_colmap_Tcw)
    registered_images = read_images_binary(os.path.join(args.reg_model_dir, 'images.bin'))
    print(Fore.GREEN + "Read {} images".format(len(registered_images)))
    reg_img_nm_to_id = {}
    for k in registered_images:
        reg_img_nm_to_id[registered_images[k].name] = k

    out_fn = os.path.join(args.output_path, "pose_errors.txt")
    mis_cnt = 0
    with open(out_fn, 'w') as f:
        f.write("# image_name trans_e_m rot_e_deg\n")
        for idx, img_nm in enumerate(reg_names):
            Twc_gt = eu.colmapQtToTwc(reg_qtvec[idx][0:4], reg_qtvec[idx][4:7])
            rel_img_path = os.path.join(rel_path, img_nm)
            te = float('nan')
            re = float('nan')
            if rel_img_path in reg_img_nm_to_id:
                Twc_reg = registered_images[reg_img_nm_to_id[rel_img_path]].Twc()
                te, re = eu.calPoseError(Twc_gt, Twc_reg)
            else:
                mis_cnt += 1
            f.write('{} {} {}\n'.format(img_nm, te, re))
    print(Fore.GREEN + "{} out of {} images failed to registers.".format(mis_cnt, len(reg_names)))
