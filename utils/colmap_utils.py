#!/usr/bin/env python3

import os
import numpy as np
from shutil import copytree, rmtree, copy2
import subprocess
import shlex
import sqlite3
from tqdm import tqdm
import shutil
import torch

from .torch_matchers import mutual_nn_matcher
from .colmap_read_model import CAMERA_MODELS, CAMERA_MODEL_IDS
from .colmap_read_model import read_cameras_binary, read_images_binary, read_points3d_binary

model_bins = ['images.bin', 'cameras.bin', 'points3D.bin']
model_txts = ['images.txt', 'cameras.txt', 'points3D.txt']

cam_model_name_to_id = dict([(cm.model_name, cm.model_id) for cm in CAMERA_MODELS])

supported_custom_ftrs = ['d2-net']


class MapperOptions:
    def __init__(self):
        self.params = {
            "filter_max_reproj_error": 4,
            "filter_min_tri_angle": 1.5,
            "ba_global_max_refinements": 5,
            "ba_global_max_refinement_change": 0.0005
        }
        self.pref = "Mapper."


class SiftMatchingOptions:
    def __init__(self):
        self.params = {"max_ratio": 0.8,
                       "max_distance": 0.7,
                       "max_error": 4,
                       "min_inlier_ratio": 0.25}
        self.pref = "SiftMatching."


def getImgNameToImgIdMap(database):
    connection = sqlite3.connect(database)
    cursor = connection.cursor()

    img_nm_to_id = {}
    cursor.execute("SELECT name, image_id FROM images;")
    for row in cursor:
        img_nm_to_id[row[0]] = row[1]

    cursor.close()
    connection.close()

    return img_nm_to_id


def getCameraIds(database):
    connection = sqlite3.connect(database)
    cursor = connection.cursor()

    camera_ids = []
    cursor.execute("SELECT camera_id FROM cameras;")
    for row in cursor:
        camera_ids.append(row[0])

    cursor.close()
    connection.close()

    return camera_ids


def nextCameraId(database):
    return max(getCameraIds(database)) + 1


def getImageIds(database):
    connection = sqlite3.connect(database)
    cursor = connection.cursor()

    image_ids = []
    cursor.execute("SELECT image_id FROM images;")
    for row in cursor:
        image_ids.append(row[0])

    cursor.close()
    connection.close()

    return image_ids


def addColmapCameras(colmap_cams, database, cam_ids, prior_focal_length=False):
    connection = sqlite3.connect(database)
    cursor = connection.cursor()
    for colmap_cam, cam_id in zip(colmap_cams, cam_ids):
        cursor.execute("INSERT INTO cameras VALUES (?, ?, ?, ?, ?, ?)",
                       (cam_id, cam_model_name_to_id[colmap_cam.model],
                        colmap_cam.width, colmap_cam.height, colmap_cam.params.tostring(),
                        prior_focal_length))
    connection.commit()
    cursor.close()
    connection.close()


def setCameraIdForImages(images, cam_ids, database):
    connection = sqlite3.connect(database)
    cursor = connection.cursor()
    for img, cam_id in zip(images, cam_ids):
        cursor.execute("UPDATE images SET camera_id=? WHERE name=?;", (cam_id, img))
    connection.commit()
    cursor.close()
    connection.close()


def imgIdsToPairId(image_id1, image_id2):
    if image_id1 > image_id2:
        return 2147483647 * image_id2 + image_id1
    else:
        return 2147483647 * image_id1 + image_id2


def extractFeaturesFromImages(database, img_dir, input_imgs_rel_path, method,
                              tmp_dir='/tmp', pref=''):
    if method is None:
        extr_img_list = os.path.join(tmp_dir, pref+'extractor_img_list.txt')
        print("- write image list for extraction in {}".format(extr_img_list))
        with open(extr_img_list, 'w') as f:
            f.writelines([v+'\n' for v in input_imgs_rel_path])
        extract_cmd = ('colmap feature_extractor --database_path {} --image_path {} '
                       '--image_list_path {} ').format(database, img_dir, extr_img_list)
        subprocess.call(shlex.split(extract_cmd))
    elif method in supported_custom_ftrs:
        new_img_id_s = max(getImageIds(database)) + 1
        new_cam_id_s = max(getCameraIds(database)) + 1
        new_img_ids = list(range(new_img_id_s, new_img_id_s + len(input_imgs_rel_path)))
        new_cam_ids = list(range(new_cam_id_s, new_cam_id_s + len(input_imgs_rel_path)))
        print("New image ids: ", new_img_id_s)
        print("New camera ids: ", new_cam_id_s)
        addImagesAndCameras(database, input_imgs_rel_path, new_img_ids, new_cam_ids)
        importCustomFeatures(input_imgs_rel_path, new_img_ids, img_dir, database, method)
    else:
        assert False, 'Unknown method type {}.'.format(method)


def triangulatePointsFromModel(database, img_dir, model_dir, mapper_options):
    output_dir = os.path.join(model_dir, 'output')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    triangulation_cmd = ("colmap point_triangulator --database_path {} "
                         "--image_path {} --input_path {} --output_path {} "
                         ).format(
                             database, img_dir, model_dir, output_dir)
    for k, v in mapper_options.params.items():
        triangulation_cmd += " --{} {}".format(mapper_options.pref+k, v)

    subprocess.call(shlex.split(triangulation_cmd))
    convert_cmd = ("colmap model_converter --input_path {} "
                   "--output_path {} --output_type TXT").format(output_dir, output_dir)
    subprocess.call(shlex.split(convert_cmd))
    for v in model_txts:
        os.remove(os.path.join(model_dir, v))
    for v in (model_bins + model_txts):
        shutil.copy2(os.path.join(output_dir, v), os.path.join(model_dir, v))
    shutil.rmtree(output_dir)


def makeEmpty3DPoints(model_dir):
    point_fn = os.path.join(model_dir, 'points3D.txt')
    if os.path.exists(point_fn):
        os.remove(point_fn)
    os.mknod(point_fn)


def matchFeaturesInList(database, img_dir, match_list, method, sift_match_options):
    if method is None:
        pass
    elif method in supported_custom_ftrs:
        all_img_nm_to_id = getImgNameToImgIdMap(database)
        matchCustomFeatures(all_img_nm_to_id, img_dir, match_list,
                            database, method)
    else:
        assert False, 'Unknown method type {}.'.format(method)
    matcher_import_cmd = ("colmap matches_importer --database_path {} "
                          "--match_list_path {} --match_type pairs ").format(
                              database, match_list)
    for k, v in sift_match_options.params.items():
        matcher_import_cmd += " --{} {}".format(sift_match_options.pref+k, v)

    subprocess.call(shlex.split(matcher_import_cmd))


def importCustomFeatures(image_names, image_ids, img_dir, database, method):
    connection = sqlite3.connect(database)
    cursor = connection.cursor()

    for image_name, image_id in tqdm(zip(image_names, image_ids), total=len(image_names)):
        features_path = os.path.join(
            img_dir, '%s.%s' % (image_name, method))

        keypoints = np.load(features_path)['keypoints']
        n_keypoints = keypoints.shape[0]

        # Keep only x, y coordinates.
        keypoints = keypoints[:, : 2]
        # Add placeholder scale, orientation.
        keypoints = np.concatenate([keypoints, np.ones((n_keypoints, 1)), np.zeros(
            (n_keypoints, 1))], axis=1).astype(np.float32)

        keypoints_str = keypoints.tostring()
        cursor.execute("INSERT INTO keypoints(image_id, rows, cols, data) VALUES(?, ?, ?, ?);",
                       (image_id, keypoints.shape[0], keypoints.shape[1], keypoints_str))

    # Close the connection to the database.
    connection.commit()
    cursor.close()
    connection.close()


def addImagesAndCameras(database, names, image_ids, camera_ids):
    connection = sqlite3.connect(database)
    cursor = connection.cursor()

    for nm, img_id, cam_id in zip(names, image_ids, camera_ids):
        cursor.execute(
                    "INSERT INTO images VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (img_id, nm, cam_id, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan))
    # copy the first camera
    rows = cursor.execute("SELECT * FROM cameras")
    _, model, width, height, params, prior = next(rows)
    for cam_id in camera_ids:
        cursor.execute("INSERT INTO cameras VALUES (?, ?, ?, ?, ?, ?)",
                       (cam_id, model, width, height, params, prior))
    connection.commit()
    cursor.close()
    connection.close()


def matchCustomFeatures(img_nm_to_id, img_dir, match_list, database, method):
    connection = sqlite3.connect(database)
    cursor = connection.cursor()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    with open(match_list, 'r') as f:
        raw_pairs = f.readlines()
    print('Found {} pairs.'.format(len(raw_pairs)))
    image_pair_ids = set()
    for raw_pair in tqdm(raw_pairs, total=len(raw_pairs)):
        image_name1, image_name2 = raw_pair.strip('\n').split(' ')
        if image_name1 not in img_nm_to_id or image_name2 not in img_nm_to_id:
            print("Failed to find {} - {} in known images. Skip".format(
                image_name1, image_name2))
            continue

        features_path1 = os.path.join(img_dir, '%s.%s' % (image_name1, method))
        features_path2 = os.path.join(img_dir, '%s.%s' % (image_name2, method))

        descriptors1 = torch.from_numpy(
            np.load(features_path1)['descriptors']).to(device)
        descriptors2 = torch.from_numpy(
            np.load(features_path2)['descriptors']).to(device)
        matches = mutual_nn_matcher(
            descriptors1, descriptors2).astype(np.uint32)

        image_id1, image_id2 = img_nm_to_id[image_name1], img_nm_to_id[image_name2]
        image_pair_id = imgIdsToPairId(image_id1, image_id2)
        if image_pair_id in image_pair_ids:
            continue
        image_pair_ids.add(image_pair_id)

        if image_id1 > image_id2:
            matches = matches[:, [1, 0]]

        matches_str = matches.tostring()
        cursor.execute("INSERT INTO matches(pair_id, rows, cols, data) VALUES(?, ?, ?, ?);",
                       (image_pair_id, matches.shape[0], matches.shape[1], matches_str))

    connection.commit()
    cursor.close()
    connection.close()


def sqliteExecuteAtomic(database, cmds):
    connection = sqlite3.connect(database)
    cursor = connection.cursor()
    for cmd in cmds:
        cursor.execute(cmd)
    connection.commit()
    cursor.close()
    connection.close()


def getModelDirs(sparse_top_dir):
    model_dirs = []
    for dirpath, dirnames, filenames in os.walk(sparse_top_dir):
        if not dirnames:
            if 'points3D.bin' in filenames or 'points3D.txt' in filenames:
                model_dirs.append(dirpath)
    return sorted(model_dirs)


def readMultipleModels(model_dirs):
    all_images = []
    all_points = []
    all_cameras = []
    img_nm_to_img_id = {}
    for md in model_dirs:
        print("- processing {}...".format(md))
        img_fn = os.path.join(md, 'images.bin')
        if not os.path.exists(img_fn):
            print(" - cannot find {}, skip.".format(img_fn))
            continue
        cur_imgs = read_images_binary(img_fn)
        n_del = 0
        for img_id in cur_imgs:
            img_nm_to_img_id[cur_imgs[img_id].name] = img_id
            for prev_imgs in all_images:
                if img_id in prev_imgs:
                    del prev_imgs[img_id]
                    n_del += 1
        print(" - delete {} duplicate images from previous models".format(n_del))
        all_images.append(cur_imgs)

        cur_points = read_points3d_binary(os.path.join(md, 'points3D.bin'))
        all_points.append(cur_points)

        cur_cameras = read_cameras_binary(os.path.join(md, 'cameras.bin'))
        all_cameras.append(cur_cameras)

    print('Read {} images, {} 3D points and {} cameras'.format(
        sum([len(v) for v in all_images]), sum([len(v) for v in all_points]),
        sum([len(v) for v in all_cameras])))

    return all_images, all_points, all_cameras, img_nm_to_img_id


def writeEmptyImageList(in_fn, out_fn):
    with open(os.path.join(out_fn), 'w') as fout:
        with open(os.path.join(in_fn), 'r') as fin:
            cnt = 0
            for line in fin:
                if line.startswith('#'):
                    continue
                if cnt % 2 == 0:
                    fout.write('{}\n\n'.format(line.strip()))
                cnt += 1


def convertModel(md, output_type):
    convert_cmd = ("colmap model_converter --input_path {} "
                   "--output_path {} --output_type {}").format(md, md, output_type)
    subprocess.call(shlex.split(convert_cmd))


def cleanModel(md, clean_type):
    if clean_type == 'TXT':
        for v in model_txts:
            os.remove(os.path.join(md, v))
    elif clean_type == 'BIN':
        for v in model_bins:
            os.remove(os.path.join(md, v))
    else:
        assert False, "Clean type: {}".format(clean_type)


def copyModel(in_dir, out_dir, model_type):
    if model_type == 'TXT':
        for b in model_txts:
            copy2(os.path.join(in_dir, b), os.path.join(out_dir, b))
    elif model_type == 'BIN':
        for b in model_bins:
            copy2(os.path.join(in_dir, b), os.path.join(out_dir, b))
    else:
        assert False, "Type: {}".format(model_type)


def ftrType2Suffix(ftr_type):
    suffix = '' if ftr_type is None else '_{}'.format(ftr_type)
    return suffix


def checkWorkspaceAndGetPath(colmap_ws, ftr_type=None, img_dir='images',
                             sparse_dir='sparse', database='database'):
    assert os.path.exists(colmap_ws)

    suffix = ftrType2Suffix(ftr_type)

    base_sparse_dir = os.path.join(colmap_ws, sparse_dir+suffix)

    base_img_dir = os.path.join(colmap_ws, img_dir)

    base_db = os.path.join(colmap_ws, '{}{}.db'.format(database, suffix))

    assert os.path.isdir(base_sparse_dir) and os.path.isdir(base_img_dir) and\
        os.path.exists(base_db)

    print("Found Colmap workspace:")
    print(" - {}: {}".format('sparse dir', base_sparse_dir))
    print(" - {}: {}".format('image dir', base_img_dir))
    print(" - {}: {}".format('database', base_db))

    return base_sparse_dir, base_img_dir, base_db


def cloneColmapWorkspace(input_ws, output_ws, ftr_type=None,
                         overwrite_sparse=True, overwrite_database=False):
    print("Copying COLMAP workspace...")

    input_sparse_dir, input_img_dir, input_database = checkWorkspaceAndGetPath(input_ws, ftr_type)
    suffix = ftrType2Suffix(ftr_type)

    img_dir = os.path.join(output_ws, 'images')
    sparse_dir = os.path.join(output_ws, 'sparse'+suffix)
    database = os.path.join(output_ws, 'database{}.db'.format(suffix))
    if os.path.exists(output_ws):
        print("- Output workspace exist.")
    else:
        os.makedirs(output_ws)

    if not os.path.exists(img_dir):
        copytree(input_img_dir, img_dir)
    else:
        print("- Image folder exists, skip.")

    if os.path.exists(sparse_dir) and overwrite_sparse:
        print("- Remove old sparse dir.")
        rmtree(sparse_dir)
    if not os.path.exists(sparse_dir):
        copytree(input_sparse_dir, sparse_dir)

    if not os.path.exists(database) or overwrite_database:
        print("- Copy database.")
        copy2(input_database, database)


def getKeypointsImgId(cursor, img_id, n_col=6):
    # https://github.com/colmap/colmap/blob/dev/src/base/database.cc#L54
    # The keypoints are stored as 6 columns anyway
    cursor.execute("SELECT data FROM keypoints WHERE image_id=?;",
                   (img_id,))
    row = next(cursor)
    assert row[0]
    return np.frombuffer(row[0], dtype=np.float32).reshape(-1, n_col)


def getTwoViewGeoMatchIndices(cursor, pair_id):
    cursor.execute("SELECT data from two_view_geometries where pair_id=?;",
                   (pair_id, ))
    row = next(cursor)
    assert row[0]
    return np.frombuffer(row[0], dtype=np.uint32).reshape(-1, 2)


def getCameraIdImageId(cursor, img_id):
    cursor.execute("SELECT camera_id FROM images WHERE image_id=?;",
                   (img_id,))
    row = next(cursor)
    assert row[0]
    return row[0]


def getCameraWidthAndHeight(cursor, cam_id):
    cursor.execute("SELECT width FROM cameras WHERE camera_id=?;",
                   (cam_id,))
    row = next(cursor)
    w = row[0]

    cursor.execute("SELECT height FROM cameras WHERE camera_id=?;",
                   (cam_id,))
    row = next(cursor)
    h = row[0]

    return w, h


def getCameraPrincipalPoint(cursor, cam_id):
    cursor.execute("SELECT model FROM cameras WHERE camera_id=?;",
                   (cam_id,))
    row = next(cursor)
    assert row[0]
    model_name = CAMERA_MODEL_IDS[row[0]].model_name
    print("Camera model: {}".format(model_name))

    cursor.execute("SELECT params FROM cameras WHERE camera_id=?;",
                   (cam_id,))
    row = next(cursor)
    assert row[0]
    if model_name == 'PINHOLE':
        params = np.frombuffer(row[0], dtype=np.float64).reshape(4,)
        return params[2], params[3]
    elif model_name == 'SIMPLE_RADIAL':
        params = np.frombuffer(row[0], dtype=np.float64).reshape(4,)
        return params[1], params[2]
    else:
        assert False


def getDescriptorsImgId(cursor, img_id):
    cursor.execute("SELECT data FROM descriptors WHERE image_id=?;",
                   (img_id,))
    row = next(cursor)
    assert row[0]
    return np.frombuffer(row[0], dtype=np.uint8).reshape(-1, 128)


def affineToThetaScale(affine):
    # see https://github.com/colmap/colmap/blob/dev/src/feature/types.cc
    assert affine.shape == (4,)
    scale = np.sqrt(affine[0]**2 + affine[1]**2)
    theta = np.arccos(affine[0] / scale)
    return theta, scale
