#!/usr/bin/env python3

import os
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_points_txt', type=str,
                        help="3D points in TXT format")
    parser.add_argument('--pref', type=str, default="stripped")
    parser.add_argument('--max_error', type=float, default=2.0)
    parser.add_argument('--max_z', type=float, default=None)
    parser.add_argument('--min_n_views', type=int, default=2)
    args = parser.parse_args()

    assert os.path.exists(args.input_points_txt)
    input_nm = os.path.basename(args.input_points_txt)
    input_dir = os.path.dirname(args.input_points_txt)
    assert input_nm.endswith('txt')
    out_nm = args.pref + '_' + input_nm

    print("Extract 3D points info. from {} to {}.".format(input_nm, out_nm))

    pt_cnt = 0
    with open(os.path.join(input_dir, out_nm), 'w') as fout:
        with open(args.input_points_txt, 'r') as fin:
            for line in fin:
                if line.startswith('#'):
                    continue
                elems = line.strip().split(' ')
                if float(elems[7]) > args.max_error:
                    continue
                if args.max_z and float(elems[3]) > args.max_z:
                    continue
                assert len(elems) % 2 == 0
                n_track = int((len(elems) - 8) / 2)
                if n_track <= args.min_n_views:
                    continue
                fout.write('{}\n'.format(' '.join(elems[1:4])))
                pt_cnt += 1
    print("Written {} points.".format(pt_cnt))
