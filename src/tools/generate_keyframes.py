import argparse
import codecs as cs

import os
import json
import random
import numpy as np


def generate_keyframes(data_dir, val_file, test_file, save_path, num_keyframes, unit_length=4):
    def _get_coin_len(_motion_len, _coin):
        if _coin == "double":
            new_m_length = (_motion_len // unit_length - 1) * unit_length
        elif _coin == "single":
            new_m_length = (_motion_len // unit_length) * unit_length
        return new_m_length

    def _go(file_name, _key_frames):
        with open(file_name) as fr:
            for motion_id in fr.readlines():
                motion_id = motion_id.strip()
                motion_len = len(np.load(f"{data_dir}/new_joint_vecs/{motion_id}.npy"))

                flag = False
                with cs.open(f"{data_dir}/texts/{motion_id}.txt") as f:
                    for idx, line in enumerate(f.readlines()):
                        line_split = line.strip().split("#")
                        f_tag = float(line_split[2])
                        to_tag = float(line_split[3])
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag

                        if f_tag == 0.0 and to_tag == 0.0:
                            flag = True
                        else:
                            n_motion_len = min(motion_len, int(to_tag * 20)) - int(f_tag * 20)
                            _len = _get_coin_len(n_motion_len, "double")
                            _key_frames[f"{motion_id}_{idx}_double"] = \
                                (np.random.permutation(_len)[:num_keyframes].tolist(), _len)
                            _len = _get_coin_len(n_motion_len, "single")
                            _key_frames[f"{motion_id}_{idx}_single"] = \
                                (np.random.permutation(_len)[:num_keyframes].tolist(), _len)
                if flag:
                    _len = _get_coin_len(motion_len, "double")
                    _key_frames[f"{motion_id}_double"] = \
                        (np.random.permutation(_len)[:num_keyframes].tolist(), _len)
                    _len = _get_coin_len(motion_len, "single")
                    _key_frames[f"{motion_id}_single"] = \
                        (np.random.permutation(_len)[:num_keyframes].tolist(), _len)


    key_frames = {}
    _go(val_file, key_frames)
    _go(test_file, key_frames)
    json.dump(key_frames, open(save_path, "w"), ensure_ascii=True, indent=4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--num_keyframes", type=int, default=8)
    parser.add_argument("--keyframe_type", type=str, default="random")
    args = parser.parse_args()

    assert args.keyframe_type == "random"

    seed = 1
    random.seed(seed)  # Python
    np.random.seed(seed)  # cpu vars

    generate_keyframes(args.data_dir,
                       val_file=f"{args.data_dir}/val.txt",
                       test_file=f"{args.data_dir}/test.txt",
                       save_path=f"{args.data_dir}/val_test_keyframes.txt",
                       num_keyframes=args.num_keyframes)
    print("Done!")



if __name__ == "__main__":
    main()