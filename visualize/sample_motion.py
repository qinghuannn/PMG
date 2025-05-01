from typing import Any, Dict, List, Tuple

import hydra
import rootutils
import numpy as np
import torch
import lightning.pytorch as L
from lightning.pytorch import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

import json
import os



rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #

from src.utils import (
    RankedLogger,
    extras,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)

log = RankedLogger(__name__, rank_zero_only=True)

import os.path as osp
import codecs as cs
from tqdm import tqdm

from src.data.humanml.scripts.motion_process import recover_from_ric
from src.data.humanml.utlis import convert_motion_representation, scale_data, motion_representation_to_joint


def pre_process(cfg):
    data_dir = os.path.join(cfg.data_dir, "new_joint_vecs")
    text_dir = os.path.join(cfg.data_dir, "texts")
    mean = np.load(osp.join(cfg.data_dir, "PMG_Mean.npy"))
    std = np.load(osp.join(cfg.data_dir, "PMG_Std.npy"))
    # mean = torch.from_numpy(mean)
    # std = torch.from_numpy(std)
    # unit_length = 4
    name = cfg.sample_id

    info = {"texts": []}
    motion = np.load(osp.join(data_dir, name + ".npy"))
    motion = convert_motion_representation(torch.from_numpy(motion), "hml3d", "pmg")
    motion = motion.numpy()
    motion = (motion - mean) / std

    # motion2 = np.load(osp.join(data_dir, "000268.npy"))
    # motion[12] = motion2[0]

    info["motion"] = motion
    info["name"] = name
    with (cs.open(osp.join(text_dir, name + ".txt")) as f):
        for j, line in enumerate(f.readlines()):
            line_split = line.strip().split("#")
            info["texts"].append(line_split[0])
            f_tag = float(line_split[2])
            to_tag = float(line_split[3])
            f_tag = 0.0 if np.isnan(f_tag) else f_tag
            to_tag = 0.0 if np.isnan(to_tag) else to_tag
            if f_tag == 0.0 and to_tag == 0.0:
                pass
            else:
                print("sample: %s, f_tag: %.2f, to_tag: %.2f" % (name, f_tag, to_tag))
                break

    return info, mean, std


def tf2m_generation(cfg, model):
    motion_info, mean, std = pre_process(cfg)
    log.info("Preprocssing done. Start generating!")
    if not os.path.exists(cfg.save_path):
        os.mkdir(cfg.save_path)
    if cfg.device != "cpu":
        model.to(f"cuda:{cfg.device}")
    model.eval()
    model.denoiser.eval()
    model.ema_denoiser.model.eval()
    mean = torch.from_numpy(mean).to(model.device)
    std = torch.from_numpy(std).to(model.device)

    # _given_keyframe = [int(_) for _ in x for x in cfg.keyframe.split("-")]
    given_keyframe = [[int(_) for _ in x.split(",")] for x in cfg.keyframe.split("-")]

    name = motion_info["name"]
    real_motion = motion_info["motion"]
    lens = []
    raw_keyframes = []
    keyframes = []
    texts = []
    new_name = []
    for k in given_keyframe:
        for i, text in enumerate(motion_info["texts"]):
            for j in range(cfg.repeats):
                raw_keyframes.append(k)
                motion_mask = torch.zeros(real_motion.shape[0], dtype=torch.long)
                keyframe = motion_mask.scatter(-1, torch.tensor(k), 1)
                keyframes.append(keyframe)
                texts.append(text)
                lens.append(real_motion.shape[0])
                new_name.append(f"{name}_text{i}_kf{'-'.join(str(x) for x in k)}_{j:02d}")

    real_motion = torch.from_numpy(real_motion).unsqueeze(0)
    real_motion = real_motion.repeat([len(texts), 1, 1]).to(model.device)
    lens = torch.tensor(lens, dtype=torch.long, device=model.device)
    keyframes = torch.stack(keyframes, dim=0).to(model.device)
    gen_motions, _ = model.sample_motion(real_motion, texts, lens, keyframes)
    gen_motions = gen_motions * std + mean
    # gen_joints = motion_representation_to_joint(gen_motions, joint_num=22, motion_repr="pmg").cpu().numpy()
    gen_joints = motion_representation_to_joint(gen_motions, joint_num=22, motion_repr="pmg", recover_root_from_hml3d=True).cpu().numpy()
    print(" ".join(new_name))
    for i in range(len(gen_joints)):
        np.save(osp.join(cfg.save_path, new_name[i] + ".npy"), gen_joints[i])


def mi_generation(cfg, model):
    motion_info, mean, std = pre_process(cfg)
    log.info("Preprocssing done. Start generating!")
    if not os.path.exists(cfg.save_path):
        os.mkdir(cfg.save_path)
    if cfg.device != "cpu":
        model.to(f"cuda:{cfg.device}")
    model.eval()
    model.denoiser.eval()
    model.ema_denoiser.model.eval()
    mean = torch.from_numpy(mean).to(model.device)
    std = torch.from_numpy(std).to(model.device)

    name = motion_info["name"]
    real_motion = motion_info["motion"]
    texts = []
    new_name = []
    real_motion = np.concatenate([np.zeros_like(real_motion), real_motion], axis=0)[:196]

    real_motion_len = real_motion.shape[0]
    keyframes = torch.zeros([real_motion_len], dtype=torch.long)
    for cur_range in cfg.time_range.split("-"):
        st, ed = [int(real_motion_len*float(_)) for _ in cur_range.split(",")]
        ed = min(ed+1,real_motion_len)
        keyframes[st:ed] = 1
    for i, text in enumerate(["a person sits down and then stands back up and walks forward.",
                              "a person picked up a phone and walks forward.",
                              ]):
        for j in range(cfg.repeats):
            texts.append(text)
            new_name.append(f"{name}_walk_text{i}_range{cfg.time_range}_{j:02d}")

    real_motion = torch.from_numpy(real_motion).unsqueeze(0)
    real_motion = real_motion.repeat([len(texts), 1, 1]).to(model.device)
    lens = torch.zeros([len(texts)], dtype=torch.long, device=model.device) + real_motion_len
    keyframes = keyframes.unsqueeze(dim=0).repeat([len(texts), 1]).to(model.device)
    gen_motions, _ = model.sample_motion(real_motion, texts, lens, keyframes)
    gen_motions = gen_motions * std + mean
    gen_joints = motion_representation_to_joint(gen_motions, joint_num=22, motion_repr="pmg", recover_root_from_hml3d=True).cpu().numpy()
    print(" ".join(new_name))
    for i in range(len(gen_joints)):
        np.save(osp.join(cfg.save_path, new_name[i] + ".npy"), gen_joints[i])

@task_wrapper
def sample_motion(cfg: DictConfig) -> None:
    """Evaluates given checkpoint on a datamodule testset.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Tuple[dict, dict] with metrics and dict with all instantiated objects.
    """
    assert cfg.ckpt_path

    torch.set_float32_matmul_precision('high')

    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    # log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    # datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    state_dict = torch.load(cfg.ckpt_path, map_location="cpu")["state_dict"]
    keys_list = list(state_dict.keys())
    for key in keys_list:
        if 'orig_mod.' in key:
            deal_key = key.replace('_orig_mod.', '')
            state_dict[deal_key] = state_dict[key]
            del state_dict[key]
    model.load_state_dict(state_dict, strict=False)
    
    os.makedirs(cfg.save_path, exist_ok=True)

    log.info("Starting sampling!")
    if cfg.gen_type == "tf2m":
        tf2m_generation(cfg, model)
    elif cfg.gen_type == "mi":
        mi_generation(cfg, model)
    else:
        raise ValueError(f"gen_type: {cfg.gen_type} not supported!")

    return {}, None


@hydra.main(version_base="1.3", config_path="../configs", config_name="sample_motion.yaml")
def main(cfg: DictConfig) -> None:
    """Main entry point for evaluation.

    :param cfg: DictConfig configuration composed by Hydra.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    sample_motion(cfg)


if __name__ == "__main__":
    main()
