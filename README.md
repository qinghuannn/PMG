<!-- # Progressive Human Motion Generation Based on Text and Few Motion Frames (TCSVT 2025) -->

# [Progressive Human Motion Generation Based on Text and Few Motion Frames](https://github.com/qinghuannn/PMG)  (TCSVT 2025)

[![arXiv](https://img.shields.io/badge/arXiv-<2503.13300>-<COLOR>.svg)](https://arxiv.org/abs/2503.13300)


The official PyTorch implementation of the paper [**"Progressive Human Motion Generation Based on Text and Few Motion Frames"**](https://arxiv.org/abs/2503.13300).


If you find this project or the paper useful in your research, please cite us:

```bibtex
@article{pmg,
  title={Progressive Human Motion Generation Based on Text and Few Motion Frames},
  author={Zeng, Ling-An and Wu, Gaojie and Wu, Ancong and Hu, Jian-Fang and Zheng, Wei-Shi},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  year={2025},
  publisher={IEEE}
}
```

## Getting Started

<details>

### 1. Create Conda Environment

<details>

We tested our code using Python 3.10.13, PyTorch 2.2.2, CUDA 12.1, and NVIDIA RTX 3090 GPUs.

```bash
conda create -n pmg python==3.10.13 -y
conda activate pmg

# install pytorch
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118


# install requirements
pip install -r requirements.txt

# mannuly install some dependencies
pip install diffusers==0.29.0
pip install huggingface-hub==0.23.2
```

</details>

### 2. Download and preprocess the datasets

<details>

#### 2.1 Download the datasets

We conduct experiments on the HumanML3D and KIT-ML datasets. For both datasets, you can download them by following the instructions in [HumanML3D](https://github.com/EricGuo5513/HumanML3D.git).

Then, copy both datasets to our repository. For example, the file directory for HumanML3D should look like this:

```bash
./data/HumanML3D/
├── new_joint_vecs/
├── texts/
├── Mean.npy # same as in [HumanML3D](https://github.com/EricGuo5513/HumanML3D) 
├── Std.npy # same as in [HumanML3D](https://github.com/EricGuo5513/HumanML3D) 
├── train.txt
├── val.txt
├── test.txt
├── train_val.txt
└── all.txt
```

#### 2.2 Prepare the other necessary datas

Our model requires additional files for training and testing. Please download them from [here](https://1drv.ms/u/c/76593cf7b7fc849c/EQ9AHbfIs91LpvfU0DE4B2wBoGRa-FglCXrq6uhEJVcsmw?e=dYYcC5). Please extract the file and move them to the 'data' folder. Then, the 'data' directory should look like this:
```bash
./data
├── HumanML3D
│   ├── label.xlsx
│   ├── PMG_Mean.npy
│   ├── PMG_Std.npy
│   ├── val_test_keyframes.txt
│   └── ... 
└── KIT-ML
    ├── PMG_Mean.npy
    ├── PMG_Std.npy
    ├── val_test_keyframes.txt
    └── ...
```

</details>



### 3. Download Dependencies and Pretrained Models

<details>

Download and unzip dependencies from [here](https://1drv.ms/u/c/76593cf7b7fc849c/EVh1Hfplyk9Bm9wU4iCpjisB_9RpQCtoOBVZE6_Bqr3SMw?e=8WQfAM).

Download and unzip pretrained models from [here](https://1drv.ms/u/c/76593cf7b7fc849c/EbUiF2aQb9NEo_sPdEi51fEBM8gSnEODn4g4qTh6vGa0sw?e=Ps3URO).

Then, the file directory should look like this:

```bash
./
├── checkpoints
│   ├── hml3d.ckpt
│   ├── kit.ckpt
│   └── kit_new.ckpt
├── deps
│   ├── glove
│   ├── Mobert
│   ├── MotionCLIP
│   ├── render_deps
│   └── t2m_guo
└── ...
```

</details>


</details>

## Training

<details>

We train our PMG on two RTX 3090 GPUs.

- **HumanML3D**
```bash
python src/train.py trainer=ddp2 trainer.devices=\"2,3\" logger=wandb data=hml3d\
  callbacks/model_checkpoint=fid \
  data.batch_size=256 data.repeat_dataset=20 \
  data.keyframe_info.num=2 data.keyframe_info.type=random1 model.k_step=3 \
  trainer.max_epochs=250 trainer.precision=16-mixed \
  model.noise_scheduler.prediction_type=sample\
  model=pmg model.mask_prior=1e-3 \
  model.compile=true task_name=hml3d_train
```

- **KIT-ML**
```bash
python src/train.py trainer=ddp2 trainer.devices=\"0,1\" logger=wandb \
  data=kit callbacks/model_checkpoint=fid \
  data.repeat_dataset=100 data.batch_size=256 trainer.max_epochs=200 \
  data.keyframe_info.num=2 data.keyframe_info.type=random1 \
  model.k_step=3 trainer.precision=16-mixed \
  model=pmg \
  model.compile=true task_name=nkit_train
```




</details>

## Evaluation

<details>

Set ```model.metrics.enable_mm_metric``` to ```True``` to evaluate Multimodality. Setting ```model.metrics.enable_mm_metric``` to ```False``` can speed up the evaluation. The ```data.keyframe_info.num``` is used to control the number of specified keyframes.

- **HumanML3D**
```bash
python src/eval.py trainer=gpu trainer.devices=\"6,\" logger=tensorboard \
  data=hml3d data.test_batch_size=256 \
  data.keyframe_info.num=1 data.keyframe_info.type=random \
  model=pmg model.k_step=3 model.guidance_scale=3 \
  model.noise_scheduler.prediction_type=sample model.step_num=10 \
  task_name=hml_eval model.metrics.replicate_times=20 model.metrics.enable_mm_metric=false \
  ckpt_path=./checkpoints/pmg-hml3d.ckpt 
```

- **KIT-ML**

```bash
python src/eval.py trainer=gpu trainer.devices=\"3,\" logger=tensorboard \
  data=kit data.test_batch_size=256\
  data.keyframe_info.num=1 data.keyframe_info.type=random \
  model=pmg model.k_step=3 model.guidance_scale=3 model.step_num=10 \
  task_name=new_kit_eval model.metrics.replicate_times=20 model.metrics.enable_mm_metric=false\
  ckpt_path=./checkpoints/pmg-kit.ckpt 
```

- **HumanML3D-Sub**


```bash
python src/eval.py trainer=gpu trainer.devices=\"4,\" logger=tensorboard \
  data=hml3d_keyframe data.test_batch_size=256 \
  data.keyframe_info.num=2 data.keyframe_info.type=gt \
  model=pmg model.k_step=3 model.guidance_scale=3 \
  model.noise_scheduler.prediction_type=sample model.step_num=10 \
  task_name=hml_kf_eval model.metrics.replicate_times=20 model.metrics.enable_mm_metric=false \
  ckpt_path=./checkpoints/pmg-hml3d.ckpt
```

</details>


## Motion Generation

<details>

Since our PMG is a text-frame-to-motion method, generation necessitates both text input and given frames. Consequently, we provide an example command below that utilizes frames sourced from the dataset to fulfill the role of these given frames:

```bash
python visualize/sample_motion.py trainer.devices=\'4,\' model.noise_scheduler.prediction_type=sample\
  model.k_step=3 repeats=10 gen_type=tf2m sample_id=000534 keyframe=\"25,56\"

```

</details>


## Visualization

<details>


```bash
CUDA_VISIBLE_DEVICES=5 python -W ignore visualize/blend_render_smooth.py  --mode video \
  --down_sample 1 --motion_list 000534_text0_kf25-56_00 000534_text0_kf25-56_01  --disable_floor --regen

```

</details>

## Training and Evaluation of MotionCLIP

<details>
We train our MotionCLIP on one RTX 3090 GPU using the following command:

- **HumanML3D**
```bash
python src/train.py trainer.devices=\'0,\' data=hml3d logger=wandb \
  data.batch_size=256 trainer.max_epochs=30 data.repeat_dataset=1 \
  data.motion_repr=hml3d data.motion_dim=263 \
  model.only_train_mtenc=10 callbacks/model_checkpoint=motion_clip \
  trainer.deterministic=true \
  trainer.strategy=ddp_find_unused_parameters_true trainer.check_val_every_n_epoch=1
```

- **KIT-ML**
```bash
python src/train.py --config-name=train_motionclip trainer.devices=\'1,\' data=kit logger=wandb \
  data.batch_size=256 trainer.max_epochs=30 data.repeat_dataset=1 \
  data.motion_repr=hml3d data.motion_dim=251 \
  model.only_train_mtenc=5 callbacks/model_checkpoint=motion_clip \
  trainer.deterministic=true \
  trainer.strategy=ddp_find_unused_parameters_true trainer.check_val_every_n_epoch=1
```

The detailed evaluation results for MotionCLIP are available within the PMG evaluation results. The following command is used to evaluate specific aspects of MotionCLIP's performance:

- **HumanML3D**
```bash
python src/motionclip_eval.py  trainer.devices=\"5,\" logger=tensorboard \
  data=hml3d data.test_batch_size=32 data.motion_repr=hml3d data.motion_dim=263 \
  seed=0 ckpt_path=./deps/MotionCLIP/hml3d/train.ckpt
```

- **KIT-ML**
```bash
python src/motionclip_eval.py  trainer.devices=\"5,\" logger=tensorboard \
  data=kit data.test_batch_size=32 data.motion_repr=hml3d data.motion_dim=251 \
  seed=0 ckpt_path=./deps/MotionCLIP/kit/train.ckpt
```

</details>

## Citation
If you find this project or the paper useful in your research, please cite us:

```bibtex
@article{pmg,
  title={Progressive Human Motion Generation Based on Text and Few Motion Frames},
  author={Zeng, Ling-An and Wu, Gaojie and Wu, Ancong and Hu, Jian-Fang and Zheng, Wei-Shi},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  year={2025},
  publisher={IEEE}
}
```

## Acknowlegements
Thanks to all open-source projects and libraries that supported our research:

[T2M](https://github.com/EricGuo5513/text-to-motion),
[MLD](https://github.com/ChenFengYe/motion-latent-diffusion/tree/main), 
[T2M-GPT](https://github.com/Mael-zys/T2M-GPT), 
[TEMOS](https://github.com/Mathux/TEMOS),
[FLAME](https://github.com/kakaobrain/flame),
[MoMask](https://github.com/EricGuo5513/momask-codes),
[ReMoDiffuse](https://github.com/mingyuan-zhang/ReMoDiffuse)


## License
This project is licensed under the [MIT License](https://github.com/EricGuo5513/momask-codes/tree/main?tab=MIT-1-ov-file#readme).

Note that our code depends on other libraries, including SMPL, SMPL-X, PyTorch3D, and uses datasets which each have their own respective licenses that must also be followed.


