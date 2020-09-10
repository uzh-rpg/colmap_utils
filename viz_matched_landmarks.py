#!/usr/bin/env python3

import argparse
import numpy as np
import os
from PIL import Image, ImageDraw
from colorama import init, Fore
import subprocess
import shlex

import utils.exp_utils as eu
import utils.colmap_read_model as cr

init(autoreset=True)


def _truncateNormalize(low_v, max_v, val, out_max=255, out_min=0):
    if val <= low_v:
        return out_min
    if val >= max_v:
        return out_max

    return out_min + (out_max - out_min) * ((val - low_v) / (max_v - low_v))


def _jet(low_v, max_v, val):
    import matplotlib.pyplot as plt
    if val <= low_v:
        return plt.cm.jet(0.0)
    if val >= max_v:
        return plt.cm.jet(1.0)
    return plt.cm.jet((val - low_v) / (max_v - low_v))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--database', required=True)
    parser.add_argument('--img_dir', required=True)
    parser.add_argument('--model_dir', required=True)
    parser.add_argument('--rel_img_list', required=True)
    parser.add_argument('--outdir', default=None)
    parser.add_argument('--ftr_radius', type=int, default=3)
    parser.add_argument('--max_z', type=float, default=40)
    parser.add_argument('--min_z', type=float, default=2)
    parser.add_argument('--make_video', action='store_true', dest='make_video')
    parser.set_defaults(make_video=False)
    args = parser.parse_args()

    eu.printDict(args.__dict__)
    outdir = os.path.join(
        args.model_dir, 'ftrs_overlay_imgs') if not args.outdir else args.outdir
    eu.createDir(outdir, remove_exist=False)
    assert os.path.exists(outdir)
    print(Fore.GREEN + "Outdir: " + outdir)

    print(Fore.GREEN + "Reading models and images... ")
    imgs_bin = os.path.join(args.model_dir, 'images.bin')
    points_bin = os.path.join(args.model_dir, 'points3D.bin')
    assert os.path.exists(imgs_bin)
    assert os.path.exists(points_bin)
    rel_img_nms = eu.readItemList(args.rel_img_list)
    rel_img_fns = [os.path.join(args.img_dir, v) for v in rel_img_nms]
    for v in rel_img_fns:
        assert os.path.exists(v)
    print("- read {} images".format(len(rel_img_nms)))
    model_imgs = cr.read_images_binary(imgs_bin)
    img_nm_to_id = {}
    for k, v in model_imgs.items():
        img_nm_to_id[v.name] = k
    model_pts = cr.read_points3d_binary(points_bin)
    print("- read {} images and {} points from the colmap model".format(
        len(model_imgs), len(model_pts)))
    suc_imgs = []
    fail_imgs = []
    for v in rel_img_nms:
        if v in img_nm_to_id:
            suc_imgs.append(v)
        else:
            fail_imgs.append(v)
    print(Fore.GREEN + "{} success images and {} failed images.".format(
        len(suc_imgs), len(fail_imgs)))

    print(Fore.GREEN + "visualizing images...")
    for rel_img_i, rel_img_nm_i in zip(rel_img_fns, rel_img_nms):
        img_nm_i = os.path.basename(rel_img_i)
        out_img_i = os.path.join(outdir, img_nm_i)
        print('{} --> {}'.format(rel_img_i, out_img_i))

        img = Image.open(rel_img_i)
        if rel_img_nm_i in fail_imgs:
            img.save(out_img_i)
            continue

        img_i = model_imgs[img_nm_to_id[rel_img_nm_i]]
        draw = ImageDraw.Draw(img)
        for ptid in img_i.point3DiD_to_kpidx:
            assert ptid != -1
            kpidx = img_i.point3DiD_to_kpidx[ptid]
            xy = img_i.xys[kpidx].ravel().tolist()
            pt = model_pts[ptid].xyz
            z = (np.dot(img_i.Rcw(), pt) + img_i.tcw())[2]
            # c = int(_truncateNormalize(args.min_z, args.max_z, z, 255, 0))
            # c = 255 - c
            c = _jet(args.min_z, args.max_z, z)
            int_rgb = (int(255 * c[0]), int(255 * c[1]), int(255 * c[2]))

            draw.rectangle([xy[0] - args.ftr_radius, xy[1] - args.ftr_radius,
                            xy[0]+args.ftr_radius, xy[1] + args.ftr_radius],
                           fill=int_rgb)
        img.save(out_img_i)

    if args.make_video:
        vid_fn = "{}/vid.mp4".format(outdir)
        if os.path.exists(vid_fn):
            print(Fore.RED + "Delete {}".format(vid_fn))
            os.remove(vid_fn)
        vid_cmd = ("ffmpeg -framerate 10 -i {}/%05d.png  "
                   "-vcodec mpeg4 -vcodec mpeg4 -b:v 80000k  {}").format(outdir, vid_fn)
        print(Fore.BLUE + vid_cmd)
        subprocess.call(shlex.split(vid_cmd))
