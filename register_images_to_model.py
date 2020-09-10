#!/usr/bin/env python3

import os
import argparse
from colorama import Fore, init
from datetime import datetime
import subprocess
import shlex
from shutil import copy2

import utils.exp_utils as eu
import utils.colmap_utils as cu
from utils.colmap_read_model import read_images_binary, read_cameras_text

init(autoreset=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('base_ws', type=str,
                        help='workspace containing the model to be registered to')

    # images to register
    parser.add_argument('--reg_name', type=str, required=True,
                        help='the name of this registration trial, will be used in unique id')
    parser.add_argument('--reg_list_fn', type=str, required=True,
                        help='images (relative to the image dir in base_ws) to register')

    # intrinsics
    parser.add_argument('--img_nm_to_colmap_cam_list', type=str, default=None,
                        help='image name (relative to the image folder) to colmap camera mapping.'
                        'None: will estimate the intrinsics as well.')

    # type of ftr
    parser.add_argument('--method', type=str, default=None,
                        help='method (feature type) for registration')

    # what to match
    parser.add_argument('--match_pair_list', type=str, default=None,
                        help='Pairs to match. If this does not exists, will match exhaustively')

    # priors
    parser.add_argument('--img_nm_to_prior_colmap_pose', type=str, default=None,
                        help='specify the pose prior for images')
    parser.add_argument('--prior_pos_thresh', type=float, default=-1)

    # optionally filter some images in base_ws
    parser.add_argument('--base_img_substr', type=str, default=None,
                        help='only match base images that contain a certain substr')

    # whether to create a new database or use a specified existing one
    parser.add_argument('--new_database', type=str, default=None)

    # images and database (optionally to specify other names)
    parser.add_argument('--img_dir', type=str, default='images')

    # registration parameters
    parser.add_argument('--min_num_inliers', type=int, default=10)

    # misc
    parser.add_argument('--upref_no_time', action='store_false', dest='upref_use_time')

    parser.set_defaults(upref_use_time=True)
    args = parser.parse_args()
    eu.printDict(args.__dict__)
    ba_refine_extra_params = 1
    ba_refine_focal_length = 1

    if args.upref_use_time:
        upref = datetime.today().strftime("%Y%m%d-%H%M%S_") + args.reg_name.replace('/', '_') + '_'
    else:
        upref = args.reg_name.replace('/', '_') + '_'
    suffix = cu.ftrType2Suffix(args.method)

    print(Fore.GREEN + "> Check base model and images to register...")
    base_sparse_dir, base_img_dir, base_db = cu.checkWorkspaceAndGetPath(args.base_ws,
                                                                         args.method,
                                                                         args.img_dir)
    model_dirs = cu.getModelDirs(base_sparse_dir)
    assert len(model_dirs) == 1, "only single model situation is supported."
    base_model_dir = model_dirs[0]
    print("Will register to model {}".format(base_model_dir))

    # check if there are images to register
    if args.reg_list_fn and os.path.exists(args.reg_list_fn):
        reg_imgs_nms = eu.readItemList(args.reg_list_fn)
        print("Found list of {} images to register.".format(len(reg_imgs_nms)))
        reg_imgs_relpath = reg_imgs_nms
    else:
        assert False, "cannot find images to register."

    print(Fore.GREEN + "> Prepare essential files and directories...")
    if args.new_database:
        print("New database specified, will not copy.")
        new_base_db = args.new_database
        assert os.path.exists(new_base_db)
    else:
        new_base_db = os.path.join(args.base_ws, '{}database{}.db'.format(upref, suffix))
        if os.path.exists(new_base_db):
            os.remove(new_base_db)
        copy2(base_db, new_base_db)

    base_tmp_dir = eu.createDir(os.path.join(args.base_ws, 'tmp_files'))
    print("Temporary files (image and match lists) will be written to {}".format(base_tmp_dir))

    all_base_images = read_images_binary(os.path.join(base_model_dir, 'images.bin'))
    base_imgs_nms = sorted([img.name for img in all_base_images.values()])
    img_nm_to_img_id = {}
    for img in all_base_images.values():
        img_nm_to_img_id[img.name] = img.id
    print("Found {} images from the base model.".format(len(base_imgs_nms)))
    if args.base_img_substr:
        base_imgs_nms = [v for v in base_imgs_nms if args.base_img_substr in v]
        print("Going to match {} that contain specified substr.".format(len(base_imgs_nms)))
    base_sparse_outdir = os.path.join(args.base_ws, upref + 'sparse' + suffix)
    print("The new model will be written to {}.".format(base_sparse_outdir))

    img_nm_to_prior_pose = None
    if args.img_nm_to_prior_colmap_pose and os.path.exists(args.img_nm_to_prior_colmap_pose):
        assert False, "position prior not implemented yet."
        assert args.prior_pos_thresh > 0, "need positive position prior value"

    print(Fore.GREEN + "Step 1: Extract features from new images.")
    cu.extractFeaturesFromImages(new_base_db, base_img_dir, reg_imgs_relpath, args.method,
                                 base_tmp_dir, upref)

    # optinally get camera intrinsics
    if args.img_nm_to_colmap_cam_list:
        cur_cam_id = cu.nextCameraId(new_base_db)
        print("new camera id starts from {}.".format(cur_cam_id))
        colmap_cameras = read_cameras_text(args.img_nm_to_colmap_cam_list, int_id=False)
        print("Found {} colmap cameras.".format(len(colmap_cameras)))
        cameras_to_add = []
        ids_to_add = []
        reg_imgs_to_delete = []
        for reg_img in reg_imgs_relpath:
            if reg_img not in colmap_cameras:
                reg_imgs_to_delete.append(reg_img)
                continue
            cameras_to_add.append(colmap_cameras[reg_img])
            ids_to_add.append(cur_cam_id)
            cur_cam_id += 1
        for img_d in reg_imgs_to_delete:
            reg_imgs_relpath.remove(img_d)
        cu.addColmapCameras(cameras_to_add, new_base_db, ids_to_add, prior_focal_length=True)
        cu.setCameraIdForImages(reg_imgs_relpath, ids_to_add, new_base_db)
        ba_refine_extra_params = 0
        ba_refine_focal_length = 0
        print("Added {} cameras to the database, till id {}.".format(
            len(cameras_to_add), cur_cam_id))
    else:
        ba_refine_extra_params = 1
        ba_refine_focal_length = 1

    print(Fore.GREEN + "Step 2: Match new images to existing images.")
    if args.match_pair_list:
        print("- found image pairs to match.")
        match_img_list = args.match_pair_list
    else:
        match_img_list = os.path.join(base_tmp_dir, upref+'matcher_img_list.txt')
        print("- write image list for matcher to {}".format(match_img_list))
        with open(match_img_list, 'w') as f:
            for reg_idx, reg_img_i in enumerate(reg_imgs_relpath):
                for base_img_i in base_imgs_nms:
                    if img_nm_to_prior_pose:
                        assert False, "Prior filtering not implemented yet"
                    else:
                        f.write('{} {}\n'.format(reg_img_i, base_img_i))
    sift_match_options = cu.SiftMatchingOptions()
    cu.matchFeaturesInList(new_base_db, base_img_dir, match_img_list, args.method,
                           sift_match_options)

    eu.createDir(base_sparse_outdir)
    eu.dumpArgsAsJson(base_sparse_outdir, args)
    print(Fore.GREEN + "Step 3: Localize new images.")
    registration_cmd = ("colmap image_registrator --database_path {} "
                        "--Mapper.abs_pose_min_num_inliers {} "
                        "--Mapper.ba_refine_focal_length {} "
                        "--Mapper.ba_refine_extra_params {} "
                        "--input_path {} --output_path {}").format(
                            new_base_db, args.min_num_inliers,
                            ba_refine_focal_length, ba_refine_extra_params,
                            base_model_dir, base_sparse_outdir)
    subprocess.call(shlex.split(registration_cmd))
    cu.convertModel(base_sparse_outdir, 'TXT')
