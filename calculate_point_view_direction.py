#!/usr/bin/env python3

import os
import argparse
from tqdm import tqdm
import numpy as np

from colorama import init, Fore

from utils.colmap_read_model import read_images_text, read_points3D_text, read_points3D_id_text


init(autoreset=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_dir', type=str,
                        help="model directory that contains BIN model")
    parser.add_argument('--out_fn', type=str, default='view_direction.txt',
                        help='the view direction corresponding of 3D points')
    parser.add_argument('--max_error', type=float, default=2.0)
    parser.add_argument('--max_z', type=float, default=None)
    parser.add_argument('--min_n_views', type=int, default=2)
    args = parser.parse_args()

    points_fn = os.path.join(args.model_dir, 'points3D.txt')
    images_fn = os.path.join(args.model_dir, 'images.txt')
    out_fn = os.path.join(args.model_dir, args.out_fn)
    assert os.path.exists(points_fn)
    assert os.path.exists(images_fn)

    print(Fore.YELLOW + "Cal. mean view direction from {} and {} --> {}".format(
        points_fn, images_fn, out_fn))

    images = read_images_text(images_fn)
    points = read_points3D_text(points_fn)
    point_ids = read_points3D_id_text(points_fn)
    assert len(point_ids) == len(points), "{} vs {}".format(len(points_fn), len(points))
    print("Read {} images and {} points.".format(len(images), len(points)))

    cnt = 0
    with open(out_fn, 'w') as f:
        for pid in tqdm(point_ids):
            point = points[pid]

            if point.error > args.max_error:
                continue
            if len(point.image_ids) <= args.min_n_views:
                continue
            if args.max_z and point.xyz[2] > args.max_z:
                continue

            img_pos = np.zeros((len(point.image_ids), 3))
            for img_idx, img_id in enumerate(point.image_ids):
                img_pos[img_idx] = images[img_id].twc()
            directions = img_pos - point.xyz
            directions = directions / np.linalg.norm(directions, ord=2, axis=1, keepdims=True)
            aver_dir = np.mean(directions, axis=0)
            aver_dir = aver_dir / np.linalg.norm(aver_dir)
            f.write('{}\n'.format(' '.join([str(v) for v in aver_dir.tolist()])))
            cnt += 1
    print(Fore.GREEN + "Written {} points.".format(cnt))
