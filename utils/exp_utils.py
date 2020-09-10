#!/usr/bin/env python3

import os
import numpy as np
import json
from .colmap_read_model import qvec2rotmat, rotmat2qvec
import shutil

img_ext = ['.png', '.jpg', ".JPG", ".PNG"]


def dumpArgsAsJson(log_dir, args):
    with open(os.path.join(log_dir, 'args.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)


def printDict(d):
    print("Arguments:\n", '\n'.join(
        ['- {}: {}'.format(k, v) for k, v in d.items()]))


def readItemList(fn):
    files = []
    with open(fn, 'r') as f:
        files = f.readlines()
    return [v.strip('\n') for v in files]


def readNamedValues(fn, separator=' '):
    names = []
    values = []
    with open(fn, 'r') as f:
        for line in f:
            items = line.strip().split(separator)
            names.append(items[0])
            values.append(list(map(float, items[1:])))

    return names, values


def readValuesList(fn, nval, separator=' ', sort=False):
    names = []
    values = []
    with open(fn, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            items = line.strip().split(separator)
            if len(items) == nval+1:
                names.append(items[0])
                values.append(list(map(float, items[1:])))
            elif len(items) == nval:
                values.append(list(map(float, items[0:])))
            else:
                assert False, "{}".format(len(items))
    if sort and names:
        names, values = (list(t) for t in zip(*sorted(zip(names, values))))
    if not names:
        names = [None] * len(values)

    return names, values


def readStringsDict(fn, separator=' '):
    nm_to_str = {}
    with open(fn, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            items = line.strip().split(separator)
            assert items[0] not in nm_to_str
            nm_to_str[items[0]] = separator.join(items[1:])

    return nm_to_str


def readStringsList(fn, separator=' '):
    names = []
    strings = []
    with open(fn, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            items = line.strip().split(separator)
            names.append(items[0])
            strings.append(separator.join(items[1:]))

    return names, strings


def saveValuesList(fn, names, values):
    assert len(names) == len(values)
    with open(fn, 'w') as f:
        for n, vs in zip(names, values):
            f.write("{} {}\n".format(n, ' '.join([str(v) for v in vs])))


def getDirList(top_dir, valid_dir_list=None):
    items = sorted([v for v in os.listdir(top_dir)
                    if os.path.isdir(os.path.join(top_dir, v))])
    valid_dirs = items.copy()
    print("Found {} directories under {}.".format(len(items), top_dir))
    if valid_dir_list:
        valid_dirs = readItemList(valid_dir_list)
        for v in valid_dirs:
            assert v in items
    print("Found {} valid directories.".format(len(valid_dirs)))

    return valid_dirs


def getSampleIndices(total_N, sample_N):
    step = int((total_N - 1) / (sample_N - 1))
    indices = list(range(0, total_N, step))

    assert indices[-1] < total_N

    return indices


def valueListToStrList(val_list):
    return [str(v) for v in val_list]


def npArrayToStrList(arr):
    return [str(v) for v in arr.ravel().tolist()]


def colmapQtvecToTwc(qtvec_list):
    Twc_list = []
    for qtvec in qtvec_list:
        assert len(qtvec) == 7
        Twc_list.append(colmapQtToTwc(np.array(qtvec[0:4]),
                                      np.array(qtvec[4:7])))

    return Twc_list


def colmapQtToTwc(qvec, tvec):
    Twc = np.eye(4)
    Rcw = qvec2rotmat(qvec)
    Twc[0:3, 0:3] = Rcw.transpose()
    Twc[0:3, 3] = -np.dot(Rcw.transpose(), tvec)

    return Twc


def TwcToColmapQT(Twc):
    qvec = rotmat2qvec(Twc[0:3, 0:3].transpose())
    tvec = -np.dot(Twc[0:3, 0:3].transpose(), Twc[0:3, 3])
    return qvec.tolist() + tvec.tolist()


def pathToName(path):
    return path.replace('/', '_').replace('.', '_')


def createDir(dir, remove_exist=False):
    if remove_exist:
        if os.path.exists(dir):
            shutil.rmtree(dir)

    if not os.path.exists(dir):
        os.makedirs(dir)

    return dir


def updateImageAndCameraNamesInplace(img_names, cameras):
    print("Updating names...")
    for idx in range(len(img_names)):
        nm = img_names[idx]
        if nm:
            assert nm == cameras[idx].name
            print("> img name {}".format(nm))
            nm = pathToName(nm)
        else:
            nm = "{0:05d}".format(idx)
        print("< new name {}".format(nm))
        cameras[idx].name = nm
        img_names[idx] = nm


def filterMultiple(values_list, indices):
    out = []
    for values in values_list:
        out.append([values[i] for i in indices])
    return out


def calPoseError(T_gt, T_est):
    Rwc_gt = T_gt[0:3, 0:3]
    Rwc_est = T_est[0:3, 0:3]
    dR = np.dot(Rwc_gt.transpose(), Rwc_est)
    e_rot = np.abs(np.rad2deg(np.arccos(min(1.0, 0.5 * (np.trace(dR) - 1)))))
    e_trans = np.linalg.norm((T_gt[0:3, 3] - T_est[0:3, 3]))

    return e_trans, e_rot


def countPoseError(t_thresh, r_thresh, t_err, r_err):
    assert len(t_thresh) == len(r_thresh)
    n_cat = len(t_thresh)
    assert len(t_err) == len(r_err)

    n_below = [0] * n_cat
    for et, er in zip(t_err, r_err):
        for idx in range(len(t_thresh)):
            if et <= t_thresh[idx] and er <= r_thresh[idx]:
                n_below[idx] += 1

    return n_below
