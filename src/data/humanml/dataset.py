import codecs as cs
import os.path
import random
import json
from os.path import join as pjoin

import numpy as np
import torch
from torch.utils import data

from .scripts.motion_process import process_file, recover_from_ric
from .scripts.word_vectorizer import WordVectorizer
from .utlis import convert_motion_representation, scale_data, motion_representation_to_joint

"""For use of training text-2-motion generative model"""

class Text2MotionDatasetV2(data.Dataset):

    def __init__(
        self,
        dataset_name,
        data_dir,
        split_file,
        w_vectorizer_path,
        max_motion_length,
        min_motion_length,
        max_text_len,
        unit_length,
        norm_method=1,
        scale=1,
        motion_repr="hml3d",
        no_aug=False,
        keyframe_info=None,
        repeat_dataset=1
    ):
        self.dataset_name = dataset_name
        self.w_vectorizer = WordVectorizer(w_vectorizer_path, "our_vab")
        self.max_length = 20
        self.pointer = 0
        self.max_motion_length = max_motion_length
        # min_motion_len = 40 if dataset_name =='t2m' else 24
        self.min_motion_length = min_motion_length
        self.max_text_len = max_text_len
        self.unit_length = unit_length
        self.repeat_dataset = repeat_dataset

        motion_dir = pjoin(data_dir, "new_joint_vecs")
        text_dir = pjoin(data_dir, "texts")

        data_dict = {}
        with cs.open(pjoin(data_dir, split_file), "r") as f:
            self.id_list = [line.strip() for line in f.readlines()]

        count = 0
        bad_count = 0
        new_name_list = []
        length_list = []
        for i, name in enumerate(self.id_list):
            try:
                motion = np.load(pjoin(motion_dir, name + ".npy"))
                if (len(motion)) < self.min_motion_length or (len(motion) >= 200):
                    bad_count += 1
                    continue
                text_data = []
                flag = False
                with (cs.open(pjoin(text_dir, name + ".txt")) as f):
                    for j, line in enumerate(f.readlines()):
                        text_dict = {}
                        line_split = line.strip().split("#")
                        caption = line_split[0]
                        # update
                        # if caption[-1] != ".":
                        #     caption += "."
                        tokens = line_split[1].split(" ")
                        f_tag = float(line_split[2])
                        to_tag = float(line_split[3])
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag

                        text_dict["caption"] = caption
                        text_dict["tokens"] = tokens
                        if f_tag == 0.0 and to_tag == 0.0:
                            flag = True
                            text_data.append(text_dict)
                        else:
                            try:
                                n_motion = motion[int(f_tag * 20):int(to_tag * 20)]
                                if (len(n_motion)) < self.min_motion_length or ((len(n_motion) >= 200)):
                                    continue
                                new_name = (random.choice("ABCDEFGHIJKLMNOPQRSTUVW") + "_" + name)
                                while new_name in data_dict:
                                    new_name = (random.choice(
                                        "ABCDEFGHIJKLMNOPQRSTUVW") + "_" +
                                                name)
                                data_dict[new_name] = {
                                    "motion": n_motion,
                                    "length": len(n_motion),
                                    "text": [text_dict],
                                    "keyframe_id": f"{name}_{j}"
                                }
                                new_name_list.append(new_name)
                                length_list.append(len(n_motion))
                            except:
                                # None
                                print(line_split)
                                print(line_split[2], line_split[3], f_tag,
                                      to_tag, name)
                                # break

                if flag:
                    data_dict[name] = {
                        "motion": motion,
                        "length": len(motion),
                        "text": text_data,
                        "keyframe_id": f"{name}"
                    }
                    new_name_list.append(name)
                    length_list.append(len(motion))
                    # print(count)
                    count += 1
                    # print(name)
            except:
                pass
        name_list, length_list = zip(
            *sorted(zip(new_name_list, length_list), key=lambda x: x[1]))

        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list
        self.reset_max_len(self.max_length)
        self.norm_method = norm_method
        self.nfeats = self.data_dict[self.name_list[0]]
        self.njoints = 22 if dataset_name == "hml3d" else 21

        self.scale = scale
        self.motion_repr = motion_repr
        self.load_mean_and_std(dataset_name, motion_repr, data_dir, scale)
        # update
        # self.is_mm = False

        self.is_train = os.path.basename(split_file) == "train.txt"
        self.keyframe_info = keyframe_info
        if "random" in self.keyframe_info.type and self.is_train is False:
            self.keyframes = json.load(open(keyframe_info.file))

    def load_mean_and_std(self, dataset_name, motion_repr, data_dir, scale):
        self.mean = np.load(pjoin(data_dir, "Mean.npy"))
        self.std = np.load(pjoin(data_dir, "Std.npy"))

        if motion_repr == "pmg":
            self.pmg_mean = np.load(pjoin(data_dir, "PMG_Mean.npy"))
            self.pmg_std = np.load(pjoin(data_dir, "PMG_Std.npy"))

        if abs(self.scale - 1) > 1e-6:
            self.mean = scale_data(dataset_name, "hml3d", self.mean, scale)
            self.std = scale_data(dataset_name, "hml3d", self.std, scale ** 2)
            self.pmg_mean = scale_data(dataset_name, motion_repr, self.pmg_mean, scale)
            self.pmg_std = scale_data(dataset_name, motion_repr, self.pmg_std, scale ** 2)

    def reset_max_len(self, length):
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        # print("Pointer Pointing at %d" % self.pointer)
        self.max_length = length

    def __len__(self):
        return self.repeat_dataset * (len(self.name_list) - self.pointer)

    def __getitem__(self, item):
        data_idx = self.pointer + item % (len(self.name_list) - self.pointer)
        data = self.data_dict[self.name_list[data_idx]]
        motion, m_length, text_list = data["motion"].copy(), data["length"], data["text"]
        # Randomly select a caption
        text_data = random.choice(text_list)
        caption, tokens = text_data["caption"], text_data["tokens"]

        if len(tokens) < self.max_text_len:
            # pad with "unk"
            tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
            sent_len = len(tokens)
            tokens = tokens + ["unk/OTHER"
                               ] * (self.max_text_len + 2 - sent_len)
        else:
            # crop
            tokens = tokens[:self.max_text_len]
            tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
            sent_len = len(tokens)
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        # Crop the motions in to times of 4, and introduce small variations
        if self.unit_length < 10:
            coin2 = np.random.choice(["single", "single", "double"])
        else:
            coin2 = "single"

        if coin2 == "double":
            m_length = (m_length // self.unit_length - 1) * self.unit_length
        elif coin2 == "single":
            m_length = (m_length // self.unit_length) * self.unit_length
        idx = random.randint(0, len(motion) - m_length)
        if not self.is_train:
            idx = 0

        if self.motion_repr != "hml3d":
            pmg_motion = convert_motion_representation(torch.from_numpy(motion), "hml3d", self.motion_repr)
            pmg_motion = pmg_motion[idx:idx + m_length].numpy()
            if abs(self.scale - 1) > 1e-6:
                pmg_motion = scale_data(self.dataset_name, self.motion_repr, pmg_motion, self.scale)
            pmg_motion = (pmg_motion - self.pmg_mean) / self.pmg_std
        motion = motion[idx:idx + m_length]
        # if self.is_train:
        #     motion = motion[idx:idx + m_length]
        # else:
        #     motion = motion[:m_length]
        "Z Normalization"
        motion = (motion - self.mean) / self.std

        # debug check nan
        if np.any(np.isnan(motion)):
            raise ValueError("nan in motion")

        if self.keyframe_info is not None:
            keyframe = self.gen_keyframes(self.name_list[data_idx], m_length, coin2)
            motion_mask = torch.zeros(m_length, dtype=torch.long)
            keyframe = motion_mask.scatter(-1, torch.from_numpy(keyframe), 1)
        else:
            keyframe = 0

        if self.motion_repr != "hml3d":
            return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, "_".join(tokens), keyframe, pmg_motion
        return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, "_".join(tokens), keyframe,

    def inv_transform(self, data, motion_repr="hml3d"):
        if motion_repr == "hml3d":
            mean = torch.tensor(self.mean).to(data)
            std = torch.tensor(self.std).to(data)
        else:
            mean = torch.tensor(self.pmg_mean).to(data)
            std = torch.tensor(self.pmg_std).to(data)
        if abs(self.scale - 1) > 1e-6:
            return scale_data(self.dataset_name, motion_repr, data * std + mean, 1.0 / self.scale)
        return data * std + mean

    def feats2joints(self, features, motion_repr="hml3d"):
        unnormed_data = self.inv_transform(features, motion_repr)
        return motion_representation_to_joint(unnormed_data, self.njoints, motion_repr)

    # def joints2feats(self, features):
    #     features = process_file(features, self.njoints)[0]
    #     mean = torch.tensor(self.mean).to(features)
    #     std = torch.tensor(self.std).to(features)
    #     features = (features - mean) / std
    #     return features

    def gen_keyframes(self, name, motion_len, coin):
        if self.keyframe_info.type == "uniform":
            return np.round(np.linspace(0, motion_len-1, num=self.keyframe_info.num,
                                       endpoint=True, dtype=np.float32)).astype(np.int64)
        elif self.keyframe_info.type == "uniform2":
            return np.round(np.linspace(0, motion_len - 1, num=self.keyframe_info.num,
                                        endpoint=True, dtype=np.float32)).astype(np.int64)
        elif "random" in self.keyframe_info.type:
            if self.is_train:
                if self.keyframe_info.type == "random":
                    return np.random.permutation(motion_len)[:self.keyframe_info.num]
                elif self.keyframe_info.type == "random1":
                    n = np.random.randint(1, self.keyframe_info.num + 1)
                    return np.random.permutation(motion_len)[:n]
                elif self.keyframe_info.type == "random2":
                    n = np.random.randint(0, self.keyframe_info.num + 1)
                    return np.random.permutation(motion_len)[:n]
            else:
                new_name = f"{self.data_dict[name]['keyframe_id']}_{coin}"
                keyframe = self.keyframes[new_name][0]
                if motion_len != self.keyframes[new_name][1]:
                    print("error", name, motion_len, self.keyframes[new_name][1], new_name)
                    # print(self.data_dict[name]["text"][0]['caption'], self.keyframes[f"{self.data_dict[name]['keyframe_id']}_{coin}"][2])
                    exit()
                return np.array(keyframe[:self.keyframe_info.num])


import pandas as pd


# using real keyframe
class Text2MotionDatasetV3(data.Dataset):
    def __init__(
        self,
        dataset_name,
        data_dir,
        split_file,
        w_vectorizer_path,
        max_motion_length,
        min_motion_length,
        max_text_len,
        unit_length,
        norm_method=1,
        scale=1,
        motion_repr="hml3d",
        no_aug=False,
        keyframe_info=None,
        repeat_dataset=1,
        flip_keyframe=True,
    ):
        self.dataset_name = dataset_name
        self.w_vectorizer = WordVectorizer(w_vectorizer_path, "our_vab")
        self.max_length = 20
        self.pointer = 0
        self.max_motion_length = max_motion_length
        # min_motion_len = 40 if dataset_name =='t2m' else 24
        self.min_motion_length = min_motion_length
        self.max_text_len = max_text_len
        self.unit_length = unit_length
        self.repeat_dataset = repeat_dataset
        self.motion_repr = motion_repr
        motion_dir = pjoin(data_dir, "new_joint_vecs")
        text_dir = pjoin(data_dir, "texts")
        self.motion_dir = motion_dir
        self.scale = scale

        self.keyframe_info = keyframe_info
        raw_keyframe_label = pd.read_excel(keyframe_info.file)
        keyframe_label = []

        data_dict = {}
        self.id_list = []
        cnt_skipped_data = 0
        for i in range(raw_keyframe_label.shape[0]):
            sample_id = "%06d" % int(raw_keyframe_label.iloc[i, 0])
            if not np.isnan(raw_keyframe_label.iloc[i, 1]):
                cnt_skipped_data += 1
                continue
            if not os.path.exists(os.path.join(motion_dir, sample_id + '.npy')):
                cnt_skipped_data += 1
                continue

            self.id_list.append(sample_id)
            keyframe_label.append([int(x) for x in raw_keyframe_label.iloc[i, 2:6]])
            if flip_keyframe:
                flip_sample_id = "M" + sample_id
                self.id_list.append(flip_sample_id)
                keyframe_label.append([int(x) for x in raw_keyframe_label.iloc[i, 2:6]])
        print("the number of skipped data:", cnt_skipped_data)

        count = 0
        bad_count = 0
        new_name_list = []
        length_list = []
        for i, name in enumerate(self.id_list):
            try:
                motion = np.load(pjoin(motion_dir, name + ".npy"))
                if (len(motion)) < self.min_motion_length or (len(motion) >= 200):
                    bad_count += 1
                    continue
                text_data = []
                flag = False
                keyframe = keyframe_label[i]
                with (cs.open(pjoin(text_dir, name + ".txt")) as f):
                    for j, line in enumerate(f.readlines()):
                        text_dict = {}
                        line_split = line.strip().split("#")
                        caption = line_split[0]

                        tokens = line_split[1].split(" ")
                        f_tag = float(line_split[2])
                        to_tag = float(line_split[3])
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag

                        text_dict["caption"] = caption
                        text_dict["tokens"] = tokens
                        if f_tag == 0.0 and to_tag == 0.0:
                            flag = True
                            text_data.append(text_dict)
                        else:
                            try:
                                idx_start, idx_end = int(f_tag * 20), int(to_tag * 20)
                                n_motion = motion[idx_start:idx_end]
                                if (len(n_motion)) < self.min_motion_length or ((len(n_motion) >= 200)):
                                    continue
                                new_name = (random.choice("ABCDEFGHIJKLMNOPQRSTUVW") + "_" + name)
                                while new_name in data_dict:
                                    new_name = (random.choice(
                                        "ABCDEFGHIJKLMNOPQRSTUVW") + "_" +
                                                name)

                                data_dict[new_name] = {
                                    "motion": n_motion,
                                    "length": len(n_motion),
                                    "text": [text_dict],
                                    "keyframe": [x for x in keyframe if idx_start <= x < idx_end]
                                }
                                new_name_list.append(new_name)
                                length_list.append(len(n_motion))
                            except:
                                # None
                                print(line_split)
                                print(line_split[2], line_split[3], f_tag,
                                      to_tag, name)
                                # break

                if flag:
                    data_dict[name] = {
                        "motion": motion,
                        "length": len(motion),
                        "text": text_data,
                        "keyframe": keyframe
                    }
                    new_name_list.append(name)
                    length_list.append(len(motion))
                    # print(count)
                    count += 1
                    # print(name)
            except:
                pass
        name_list, length_list = zip(
            *sorted(zip(new_name_list, length_list), key=lambda x: x[1]))

        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list
        self.reset_max_len(self.max_length)
        self.norm_method = norm_method
        self.nfeats = motion.shape[1]
        self.njoints = 22 if dataset_name == "hml3d" else 21
        # update
        # self.is_mm = False

        self.load_mean_and_std(dataset_name, motion_repr, data_dir, scale)

    def load_mean_and_std(self, dataset_name, motion_repr, data_dir, scale):
        self.mean = np.load(pjoin(data_dir, "Mean.npy"))
        self.std = np.load(pjoin(data_dir, "Std.npy"))

        if motion_repr == "pmg":
            self.pmg_mean = np.load(pjoin(data_dir, "PMG_Mean.npy"))
            self.pmg_std = np.load(pjoin(data_dir, "PMG_Std.npy"))

        if abs(scale - 1) > 1e-6:
            self.mean = scale_data(dataset_name, "hml3d", self.mean, scale)
            self.std = scale_data(dataset_name, "hml3d", self.std, scale ** 2)
            self.pmg_mean = scale_data(dataset_name, motion_repr, self.pmg_mean, scale)
            self.pmg_std = scale_data(dataset_name, motion_repr, self.pmg_std, scale ** 2)

    def reset_max_len(self, length):
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        # print("Pointer Pointing at %d" % self.pointer)
        self.max_length = length

    def __len__(self):
        return self.repeat_dataset * (len(self.name_list) - self.pointer)

    def __getitem__(self, item):
        data_idx = self.pointer + item % (len(self.name_list) - self.pointer)
        data = self.data_dict[self.name_list[data_idx]]
        motion, m_length, text_list = data["motion"].copy(), data["length"], data["text"]
        keyframe = data["keyframe"].copy()
        # Randomly select a caption
        text_data = random.choice(text_list)
        caption, tokens = text_data["caption"], text_data["tokens"]

        if len(tokens) < self.max_text_len:
            # pad with "unk"
            tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
            sent_len = len(tokens)
            tokens = tokens + ["unk/OTHER"
                               ] * (self.max_text_len + 2 - sent_len)
        else:
            # crop
            tokens = tokens[:self.max_text_len]
            tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
            sent_len = len(tokens)
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        # Crop the motions in to times of 4, and introduce small variations
        if self.unit_length < 10:
            coin2 = np.random.choice(["single", "single", "double"])
        else:
            coin2 = "single"

        if coin2 == "double":
            m_length = (m_length // self.unit_length - 1) * self.unit_length
        elif coin2 == "single":
            m_length = (m_length // self.unit_length) * self.unit_length
        idx = 0

        if self.motion_repr != "hml3d":
            pmg_motion = convert_motion_representation(torch.from_numpy(motion), "hml3d", self.motion_repr)
            pmg_motion = pmg_motion[idx:idx + m_length].numpy()
            if abs(self.scale - 1) > 1e-6:
                pmg_motion = scale_data(self.dataset_name, self.motion_repr, pmg_motion, self.scale)
            pmg_motion = (pmg_motion - self.pmg_mean) / self.pmg_std
        motion = motion[idx:idx + m_length]

        "Z Normalization"
        motion = (motion - self.mean) / self.std

        # debug check nan
        if np.any(np.isnan(motion)):
            raise ValueError("nan in motion")

        if self.keyframe_info.type == "gt":
            keyframe = [x - idx for x in keyframe if idx <= x < idx + m_length]
            while len(keyframe) < self.keyframe_info.num:
                idx = np.random.randint(m_length)
                if idx not in keyframe:
                    keyframe.append(idx)
            keyframe = np.array(keyframe)
            if len(keyframe) > self.keyframe_info.num:
                keyframe = np.random.choice(keyframe, self.keyframe_info.num, replace=False)
        else:
            keyframe = np.random.permutation(m_length)[:self.keyframe_info.num]


        motion_mask = torch.zeros(m_length, dtype=torch.long)
        keyframe = motion_mask.scatter(-1, torch.from_numpy(keyframe), 1)

        if self.motion_repr != "hml3d":
            return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, "_".join(tokens), keyframe, pmg_motion

        return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, "_".join(tokens), keyframe

    def inv_transform(self, data, motion_repr="hml3d"):
        if motion_repr == "hml3d":
            mean = torch.tensor(self.mean).to(data)
            std = torch.tensor(self.std).to(data)
        else:
            mean = torch.tensor(self.pmg_mean).to(data)
            std = torch.tensor(self.pmg_std).to(data)
        if abs(self.scale - 1) > 1e-6:
            return scale_data(self.dataset_name, motion_repr, data * std + mean, 1.0 / self.scale)
        return data * std + mean

    # update
    def feats2joints(self, features, motion_repr="hml3d"):
        unnormed_data = self.inv_transform(features, motion_repr)
        return motion_representation_to_joint(unnormed_data, self.njoints, motion_repr)
