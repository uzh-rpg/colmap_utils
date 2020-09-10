#!/usr/bin/env python3

import os
import argparse
from colorama import init, Fore
import shutil
from tqdm import tqdm

import utils.exp_utils as eu
import utils.colmap_utils as cu
from utils.colmap_read_model import read_cameras_text

init(autoreset=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('ws', type=str, help='workspace containing images to reconstruct')

    parser.add_argument('--method', type=str, default=None)

    # camera intrinsics
    parser.add_argument('--colmap_cam_txt', type=str, default=None, help='camera')
    parser.add_argument('--img_to_colmap_cam_list', type=str, default=None,
                        help='image name to camera intrinsics')
    parser.add_argument('--no_add_cam_to_db', action='store_false', dest='add_cam_to_db',
                        help='whether to add the cameras to database or just use for reconstruction')

    # poses
    parser.add_argument('--img_to_colmap_pose_list', type=str, required=True,
                        help='image name to poses (colmap convention)')

    # images
    parser.add_argument('--img_dir', type=str, default='images', help='will look for images here')
    parser.add_argument('--img_paths_nm', type=str, default='image_paths.txt',
                        help='only use the image in the text file if this file exists')

    # match
    parser.add_argument('--img_pairs_to_match', type=str, default=None)
    parser.add_argument('--match_overlap', type=int, default=-1,
                        help='sequential matcher overlap if the value is greater than 1')

    # triangulator
    parser.add_argument('--tri_filter_max_reproj', type=float, default=4.0)
    parser.add_argument('--tri_filter_min_angle', type=float, default=1.5)

    parser.add_argument('--overwrite_db', action='store_true', dest='overwrite_db')

    parser.set_defaults(overwrite_db=False, add_cam_to_db=True)
    args = parser.parse_args()
    eu.printDict(args.__dict__)

    suffix = cu.ftrType2Suffix(args.method)

    img_dir = os.path.join(args.ws, args.img_dir)
    if args.colmap_cam_txt:
        print("Found COLMAP camera file {}.".format(args.colmap_cam_txt))
        print("NOTE: this file should contain only one camera.")
    elif args.img_to_colmap_cam_list:
        print("Will use img name to camera list {}".format(args.img_to_colmap_cam_list))
        print("NOTE: this file should contain the mapping from image names to colmap intrinsics.")
    else:
        assert False, "Need to specify camera intrinsics somehow"

    print(Fore.RED + ">>> 0. Read images and poses.")

    img_fn = os.path.join(img_dir, args.img_paths_nm)
    imgs = []
    if os.path.exists(img_fn):
        print("  Read from {}".format(img_fn))
        imgs = sorted(eu.readItemList(img_fn))
        for v in imgs:
            assert os.path.exists(os.path.join(img_dir, v)), '{}'.format(os.path.join(img_dir, v))
    else:
        print("  Collection all images inside {}.".format(args.img_dir))
        imgs = sorted([v for v in os.listdir(img_dir) if os.path.splitext(v)[-1] in eu.img_ext])
    print("  Found {} images.".format(len(imgs)))

    names, qtvec_list = [], []
    if os.path.exists(args.img_to_colmap_pose_list):
        print("- Found image names - COLMAP poses mapping.")
        names, qtvec_list = eu.readValuesList(args.img_to_colmap_pose_list, 7, sort=True)
    else:
        assert False, "Cannot find pose information"
    assert len(imgs) == len(qtvec_list)
    for idx in range(len(names)):
        assert names[idx] == imgs[idx], "{} vs {}".format(names[idx],
                                                          imgs[idx])
    print(Fore.GREEN + "Read {} poses.".format(len(qtvec_list)))

    database = os.path.join(args.ws, 'database{}.db'.format(suffix))
    if os.path.exists(database) and args.overwrite_db:
        print(Fore.RED + "Remove existing database.")
        os.remove(database)

    md = os.path.join(args.ws, 'sparse{}'.format(suffix), '0')
    if os.path.exists(os.path.join(md)):
        print(Fore.RED + "Remove existing sparse dir.")
        shutil.rmtree(md)
    os.makedirs(md)

    tmp_dir = eu.createDir(os.path.join(args.ws, 'tmp_files'))

    print(Fore.RED + ">>> 1. Feature extraction")
    if args.method:
        print("Create database via standard feature extraction.")
        cu.extractFeaturesFromImages(database, img_dir, imgs, None, tmp_dir=tmp_dir)
        cu.sqliteExecuteAtomic(database,
                               ["DELETE FROM keypoints;", "DELETE FROM descriptors;"])
        img_nm_to_id = cu.getImgNameToImgIdMap(database)
        cu.importCustomFeatures(list(img_nm_to_id.keys()), list(img_nm_to_id.values()),
                                img_dir, database, args.method)
    else:
        cu.extractFeaturesFromImages(database, img_dir, imgs, None, tmp_dir=tmp_dir)

    print(Fore.RED + ">>> 2. Create empty reconstruction")
    next_cam_id = cu.nextCameraId(database)
    img_nm_to_cam_id = {}
    model_cam_txt = os.path.join(md, 'cameras.txt')
    if args.colmap_cam_txt:
        shutil.copy2(args.colmap_cam_txt, model_cam_txt)
        colmap_cameras = read_cameras_text(model_cam_txt)
        assert len(colmap_cameras) == 1
        colmap_cam = colmap_cameras[list(colmap_cameras.keys())[0]]
        assert colmap_cam.id >= next_cam_id, "{} > {}".format(colmap_cam.id, next_cam_id)
        if args.add_cam_to_db:
            cu.addColmapCameras([colmap_cam], database, [colmap_cam.id], prior_focal_length=True)
        next_cam_id = colmap_cam.id
        for v in imgs:
            img_nm_to_cam_id[v] = next_cam_id
    elif args.img_to_colmap_cam_list:
        colmap_cameras = read_cameras_text(args.img_to_colmap_cam_list, int_id=False)
        nintris = len(colmap_cameras)
        assert nintris == len(imgs)
        print("Read {} colmap cameras.".format(len(colmap_cameras)))
        with open(args.img_to_colmap_cam_list, 'r') as fintri:
            cur_cam_id = next_cam_id
            with open(model_cam_txt, 'w') as fout:
                cameras_to_add = []
                cam_ids_to_add = []
                for line in tqdm(fintri, total=nintris):
                    items = line.strip().split()
                    img_nm_to_cam_id[items[0]] = cur_cam_id
                    fout.write("{} {}\n".format(cur_cam_id, ' '.join(items[1:])))
                    cameras_to_add.append(colmap_cameras[items[0]])
                    cam_ids_to_add.append(cur_cam_id)
                    cur_cam_id += 1
            if args.add_cam_to_db:
                cu.addColmapCameras(cameras_to_add,
                                    database, cam_ids_to_add, prior_focal_length=True)
            print("Write {} cameras till the id {}.".format(len(img_nm_to_cam_id), cur_cam_id))

    else:
        assert False

    img_nm_to_img_id = cu.getImgNameToImgIdMap(database)
    with open(os.path.join(md, 'images.txt'), 'w') as f:
        for idx, img_nm in enumerate(imgs):
            qtvec_str = ' '.join(eu.valueListToStrList(qtvec_list[idx]))
            f.write("{} {} {} {}\n\n".format(
                img_nm_to_img_id[img_nm], qtvec_str, img_nm_to_cam_id[img_nm], img_nm))
    cu.makeEmpty3DPoints(md)

    print(Fore.RED + ">>> 3. Matching")
    match_list = args.img_pairs_to_match
    if not args.img_pairs_to_match:
        match_list = os.path.join(tmp_dir, 'match_list.txt')
        with open(match_list, 'w') as f:
            for idx_i, img_i in enumerate(imgs):
                range_max = len(imgs) if args.match_overlap < 0 else\
                    min(len(imgs), idx_i + args.match_overlap)
                for idx_j in range(idx_i + 1, range_max):
                    f.write("{} {}\n".format(imgs[idx_i], imgs[idx_j]))
    sift_match_options = cu.SiftMatchingOptions()
    sift_match_options.params['max_ratio'] = 0.6
    sift_match_options.params['max_distance'] = 0.5
    sift_match_options.params['max_error'] = 2
    sift_match_options.params['min_inlier_ratio'] = 0.15
    cu.matchFeaturesInList(database, img_dir, match_list, args.method, sift_match_options)

    print(Fore.RED + ">>> 4. Triangulate Points")
    mapper_opts = cu.MapperOptions()
    mapper_opts.params['filter_max_reproj_error'] = args.tri_filter_max_reproj
    mapper_opts.params['filter_min_tri_angle'] = args.tri_filter_min_angle
    cu.triangulatePointsFromModel(database, img_dir, md, mapper_opts)
