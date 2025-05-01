import os

import numpy as np
import torch
import torch.nn.functional as F
import torchmetrics
from torchmetrics import MeanMetric, MetricCollection
import lightning.pytorch as L


from diffusers import UniPCMultistepScheduler, DDPMScheduler


from .metrics import TM2TMetrics, ComputeMetrics, MMMetrics
from .nets.ema import EMAModel

from .utils.gen_prior import generate_prior_info, generate_predict_map
from .utils.utils import lengths_to_mask, replace_annotation_with_null, \
    occumpy_mem, CosineWarmupScheduler, select_partial_motion
from src.data.humanml.utlis import convert_motion_representation, mask_init_prior


class PMG(L.LightningModule):
    def __init__(self,
                 text_encoder,
                 denoiser,
                 optimizer,
                 noise_scheduler,
                 evaluator,
                 ema,
                 metrics,
                 dataset_name,
                 text_replace_prob,
                 guidance_scale,
                 step_num,
                 ocpm,
                 k_step,
                 num_keyframes,
                 mask_prior=0.1,
                 motion_repr="hml3d",
                 is_mask_init_prior=False,
                 lr_scheduler=None,
                 sample_scheduler=False,
                 **kwargs):
        super(PMG, self).__init__()
        self.save_hyperparameters(logger=False, ignore=['text_encoder', 'denoiser'])
        self.text_encoder = text_encoder
        self.denoiser = denoiser
        self.noise_scheduler = noise_scheduler
        if sample_scheduler is False:
            self.sample_scheduler = noise_scheduler
        else:
            self.sample_scheduler = sample_scheduler
        # self.sample_scheduler.set_timesteps(step_num)
        self.configure_evaluator_and_metrics(dataset_name, evaluator, metrics)
        if ema.use_ema:
            self.ema_denoiser = EMAModel(self.denoiser, ema.ema_decay)
        else:
            self.ema_denoiser = None
        self.validation_step_outputs = []

    def setup(self, stage: str) -> None:
        if self.hparams.compile and stage == "fit":
            import torch._dynamo
            torch._dynamo.config.suppress_errors = True
            self.denoiser = torch.compile(self.denoiser)

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(self.denoiser.parameters())
        if self.hparams.lr_scheduler is not None:
            lr_scheduler = self.hparams.lr_scheduler(optimizer=optimizer)
            lr_scheduler_config = {"scheduler": lr_scheduler, "interval": "epoch"}
            return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}
        return optimizer

    def configure_evaluator_and_metrics(self, dataset_name, evaluator, metrics):
        from src.models.evaluator.MotionCLIP import MotionCLIPEvaluator
        self.motionclip_evaluator = MotionCLIPEvaluator(dataset_name, evaluator.MotionCLIP_dir)

        from src.models.evaluator.T2M import T2MEvaluator
        self.t2m_evaluator = T2MEvaluator(dataset_name, evaluator.T2M_dir)

        if dataset_name == "hml3d":
            from src.models.evaluator.MoBert import MoBertEvaluator
            self.mobert_evaluator = MoBertEvaluator(dataset_name, evaluator.MoBert_dir)

        self.train_metrics = MetricCollection({"loss": MeanMetric()})
        eval_metrics = {
            "t2m": TM2TMetrics(diversity_times=300 if dataset_name == "hml3d" else 100,
                                           dist_sync_on_step=True),
            "motionclip": TM2TMetrics(diversity_times=300 if dataset_name == "hml3d" else 100,
                                           dist_sync_on_step=True),
            "CoordinateError": ComputeMetrics(njoints=22 if dataset_name == "hml3d" else 21,
                                              jointstype="humanml3d" if dataset_name == "hml3d" else "mmm",
                                              dist_sync_on_step=True)
        }
        if dataset_name == "hml3d":
            eval_metrics["mobert_alignment"] = MeanMetric(dist_sync_on_step=True)
            eval_metrics["mobert_faithfulness"] = MeanMetric(dist_sync_on_step=True)
            eval_metrics["mobert_naturalness"] = MeanMetric(dist_sync_on_step=True)
            eval_metrics["mobert_gt_alignment"] = MeanMetric(dist_sync_on_step=True)
            eval_metrics["mobert_gt_faithfulness"] = MeanMetric(dist_sync_on_step=True)
            eval_metrics["mobert_gt_naturalness"] = MeanMetric(dist_sync_on_step=True)

        if self.hparams.metrics.enable_mm_metric:
            eval_metrics["motionclip_mm_gen"] = MMMetrics(mm_num_times=self.hparams.metrics.mm_num_times, dist_sync_on_step=True)
            eval_metrics["t2m_mm_gen"] = MMMetrics(mm_num_times=self.hparams.metrics.mm_num_times, dist_sync_on_step=True)

        self.eval_metrics = MetricCollection(eval_metrics)

    def on_train_start(self):
        if self.hparams.ocpm:
            occumpy_mem(self.device.index)

    def random_mask_all_prior(self, pmap, motion_prior):
        mask = torch.rand(pmap.shape[0], dtype=torch.float, device=self.device) < self.hparams.mask_prior
        motion_prior = motion_prior * ~mask.unsqueeze(dim=-1)
        pmap[mask] = pmap[mask].masked_fill(pmap[mask]==0, 1)
        return pmap, motion_prior

    def _step_network(self, batch, batch_idx, k_cur=0):
        motion_key = "pmg_motion" if self.hparams.motion_repr != "hml3d" else "motion"
        motion, text, length, keyframes = batch[motion_key], batch["text"], batch["length"], batch["keyframes"]

        # get the text embedding for conditioning
        text = replace_annotation_with_null(text, self.hparams.text_replace_prob)
        with torch.no_grad():
            text_embed = self.text_encoder(text, self.device)

        motion_prior = keyframes
        pmap = generate_predict_map(motion_prior, length, None, k_max=self.hparams.k_step, k_cur=k_cur)
        pmap, motion_prior = self.random_mask_all_prior(pmap, motion_prior)

        if self.hparams.is_mask_init_prior:
            motion = mask_init_prior(motion, motion_prior, self.hparams.motion_repr)

        if self.hparams.ema.ema_prior != 0 and self.current_epoch >= self.hparams.ema.prior_start:
            if np.random.rand() < self.hparams.ema.prior_prob:
                assert self.ema_denoiser is not None
                motion = self.ema_predict_prior(motion, text_embed, motion_prior, pmap)


        gmotion, gpos, gmask = select_partial_motion(motion, pmap == 0)
        gmotion[gmask] = 0
        # prediction info
        pmotion, ppos, pmask = select_partial_motion(motion, pmap == 1)
        # random timestep
        timestep = torch.randint(0, self.noise_scheduler.config.num_train_timesteps,
                                 (pmotion.size(0),), device=pmotion.device).long()
        # add noise
        noise = torch.randn_like(pmotion, device=pmotion.device)
        pmotion_noise = self.noise_scheduler.add_noise(pmotion, noise, timestep)

        # print(pmotion.max(), pmotion.min())
        if torch.any(torch.isnan(pmotion)):
            print("input pmotion contains nan!")

        # predict noise (N, L, C)
        output = self.denoiser(text_embed, timestep, pmotion_noise, ppos, pmask, gmotion, gpos, gmask, motion_prior)
        # compute loss
        out = {"output": output, "noise": noise, "mask": pmask, "x_0": pmotion}
        losses = self.cal_loss(out)
        return losses
    
    @torch.no_grad()
    def ema_predict_prior(self, motion, text_embed, motion_prior, pmap):
        new_pmap = pmap.clone()
        new_pmap[new_pmap == 1] = 2
        new_pmap[(motion_prior == 1) & (new_pmap == 0)] = 0
        new_pmap[(motion_prior == 0) & (new_pmap == 0)] = 1
        gmotion, gpos, gmask = select_partial_motion(motion, new_pmap == 0)
        pmotion, ppos, pmask = select_partial_motion(motion, new_pmap == 1)
        # usage #1
        if self.hparams.ema.ema_prior == 1:
            if self.hparams.ema.step_range == -1:
                ema_range = self.noise_scheduler.config.num_train_timesteps
            else:
                ema_range = self.hparams.ema.step_range
            timestep = torch.randint(0, ema_range, (pmotion.size(0),), device=pmotion.device).long()
            noise = torch.randn_like(pmotion, device=pmotion.device)
            pmotion_noise = self.noise_scheduler.add_noise(pmotion, noise, timestep)

            pred_pmotion = self.ema_denoiser.model(text_embed, timestep, pmotion_noise, ppos, pmask,
                                                   gmotion, gpos, gmask, motion_prior)
            motion[new_pmap == 1] = pred_pmotion[~pmask].float()
        return motion.detach()

    def training_step(self, batch, batch_idx):
        losses = self._step_network(batch, batch_idx)
        self.train_metrics["loss"](losses["loss"])
        self.log(f"Train/loss", self.train_metrics["loss"], prog_bar=True)
        return losses["loss"]

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.ema_denoiser is not None and self.global_step % self.hparams.ema.ema_update == 0:
            if self.global_step <= self.hparams.ema.ema_start:
                self.ema_denoiser.set(self.denoiser)
            else:
                self.ema_denoiser.update(self.denoiser)

    def on_train_epoch_end(self):
        for key in self.train_metrics:
            self.train_metrics[key].reset()

    def validation_step(self, batch, batch_idx):
        if self.trainer.sanity_checking:
            return
        self.evaluate_model(batch)

    def on_validation_epoch_end(self):
        results = {}
        metric_output = self.eval_metrics["motionclip"].compute(sanity_flag=self.trainer.sanity_checking)
        results.update({f"Metrics/mc_{key}": value.item() for key, value in metric_output.items()})

        metric_output = self.eval_metrics["t2m"].compute(sanity_flag=self.trainer.sanity_checking)
        results.update({f"Metrics/t2m_{key}": value.item() for key, value in metric_output.items()})

        metric_output = self.eval_metrics["CoordinateError"].compute(sanity_flag=self.trainer.sanity_checking)
        results.update({f"Metrics/{key}": value.item() for key, value in metric_output.items()})

        results["Metrics/save_metric"] = (results["Metrics/t2m_R_precision_top_1"] + results["Metrics/mc_R_precision_top_1"]) /2

        if self.hparams.dataset_name == "hml3d":
            results["Metrics/mb_alignment"] = self.eval_metrics["mobert_alignment"].compute()
            results["Metrics/mb_faithfulness"] = self.eval_metrics["mobert_faithfulness"].compute()
            results["Metrics/mb_naturalness"] = self.eval_metrics["mobert_naturalness"].compute()
            results["Metrics/mb_gt_alignment"] = self.eval_metrics["mobert_gt_alignment"].compute()
            results["Metrics/mb_gt_faithfulness"] = self.eval_metrics["mobert_gt_faithfulness"].compute()
            results["Metrics/mb_gt_naturalness"] = self.eval_metrics["mobert_gt_naturalness"].compute()

        for key in self.eval_metrics:
            self.eval_metrics[key].reset()

        for k, v in metric_output.items():
            metric_output[k] = v.float()

        results.update({
            "epoch": self.trainer.current_epoch,
            "step": self.global_step,
        })
        if self.trainer.sanity_checking is False:
            self.log_dict(results, sync_dist=True)

    def on_test_start(self) -> None:
        if self.hparams.ocpm:
            if hasattr(self, "ocpm_success"):
                pass
            else:
                occumpy_mem(self.device.index)
                self.ocpm_success = True

    def test_step(self, batch, batch_idx):
        self.evaluate_model(batch, 'test')

    def on_test_epoch_end(self):
        results = {}

        if self.trainer.test_dataloaders.dataset.is_mm:
            results["Metrics/mc_mm_gen"] = self.eval_metrics["motionclip_mm_gen"].compute(sanity_flag=self.trainer.sanity_checking)["MultiModality"]
            results["Metrics/t2m_mm_gen"] = self.eval_metrics["t2m_mm_gen"].compute(sanity_flag=self.trainer.sanity_checking)["MultiModality"]
        else:
            metric_output = self.eval_metrics["motionclip"].compute(sanity_flag=self.trainer.sanity_checking)
            results.update({f"Metrics/mc_{key}": value.item() for key, value in metric_output.items()})

            metric_output = self.eval_metrics["t2m"].compute(sanity_flag=self.trainer.sanity_checking)
            results.update({f"Metrics/t2m_{key}": value.item() for key, value in metric_output.items()})

            metric_output = self.eval_metrics["CoordinateError"].compute(sanity_flag=self.trainer.sanity_checking)
            results.update({f"Metrics/{key}": value.item() for key, value in metric_output.items()})

            if self.hparams.dataset_name == "hml3d":
                results["Metrics/mb_alignment"] = self.eval_metrics["mobert_alignment"].compute()
                results["Metrics/mb_faithfulness"] = self.eval_metrics["mobert_faithfulness"].compute()
                results["Metrics/mb_naturalness"] = self.eval_metrics["mobert_naturalness"].compute()
                results["Metrics/mb_gt_alignment"] = self.eval_metrics["mobert_gt_alignment"].compute()
                results["Metrics/mb_gt_faithfulness"] = self.eval_metrics["mobert_gt_faithfulness"].compute()
                results["Metrics/mb_gt_naturalness"] = self.eval_metrics["mobert_gt_naturalness"].compute()

        for key in self.eval_metrics:
            self.eval_metrics[key].reset()

        if not self.trainer.sanity_checking:
            self.log_dict(results, sync_dist=True, rank_zero_only=True)

    def on_test_end(self):
        for key in self.eval_metrics:
            self.eval_metrics[key].reset()

    def on_save_checkpoint(self, checkpoint):
        state_dict = checkpoint['state_dict']
        remove_keys = []
        for k, v in state_dict.items():
            if 'text_encoder' in k:
                remove_keys.append(k)
        for k in remove_keys:
            del checkpoint['state_dict'][k]

    def on_load_checkpoint(self, checkpoint):
        keys_list = list(checkpoint['state_dict'].keys())
        for key in keys_list:
            if 'orig_mod.' in key:
                deal_key = key.replace('_orig_mod.', '')
                checkpoint['state_dict'][deal_key] = checkpoint['state_dict'][key]
                del checkpoint['state_dict'][key]

    @torch.no_grad()
    def sample_motion(self, gt_motion, text, length, motion_prior):
        B, L, D = gt_motion.shape

        if self.ema_denoiser is not None:
            denoiser = self.ema_denoiser.model
        else:
            denoiser = self.denoiser

        repeated_text = text.copy()
        repeated_text.extend([""] * B)
        text_embed = self.text_encoder(repeated_text, self.device)
        tmp = gt_motion.clone()
        # motion_prior[motion_prior==1] = 0
        # motion_prior[:, 0] = 1
        final_motion_gen = torch.zeros_like(gt_motion)
        if self.hparams.is_mask_init_prior:
            tmp = mask_init_prior(tmp, motion_prior, self.hparams.motion_repr)
        final_motion_gen[motion_prior==1] = tmp[motion_prior==1]
        prediction_type = self.noise_scheduler.config.prediction_type

        for k in range(1, 1 + self.hparams.k_step):
            pmap = generate_predict_map(motion_prior, length, None,
                                        k_max=self.hparams.k_step, k_cur=k)
            if torch.sum(pmap == 1) == 0:
                break

            gmotion, gpos, gmask = select_partial_motion(final_motion_gen, pmap == 0)
            # if k == 1:
            #     pmap[pmap==0] = 1
            #     gmotion = gmotion[:, :1]
            #     gpos = gpos[:, :1]
            #     gmask = gmask[:, :1]
            #     gmask[gmask==False] = False
            # gmotion, gpos, gmask = select_partial_motion(final_motion_gen, motion_prior == 1)
            # gmask[gmask==False] = True
            gmotion[gmask] = 0
            # if k == 1:
            #     gmask[gmask==False] = True
                # print(pmap[0])
                # pmap[pmap == 0] = 1
                # exit()
                # pmap[pmap==0] = 1
            # prediction info
            noise_motion = torch.randn_like(final_motion_gen, device=gt_motion.device)
            pmotion, ppos, pmask = select_partial_motion(noise_motion, pmap == 1 )
            pmotion = pmotion * self.sample_scheduler.init_noise_sigma
            self.sample_scheduler.set_timesteps(self.hparams.step_num)
            time_steps = self.sample_scheduler.timesteps.to(gt_motion.device)

            for i, t in enumerate(time_steps):
                output = denoiser(text_embed,  t.repeat([2 * B]),
                                  pmotion.repeat([2, 1, 1]), ppos.repeat([2, 1]), pmask.repeat([2, 1]),
                                  gmotion.repeat([2, 1, 1]), gpos.repeat([2, 1]), gmask.repeat([2, 1]),
                                  motion_prior.repeat([2, 1]))

                if prediction_type == "epsilon":
                    cond_eps, uncond_eps = output.chunk(2)
                elif prediction_type == "sample":
                    cond_x0, uncond_x0 = output.chunk(2)
                    cond_eps, uncond_eps = self.obtain_eps_when_predicting_x_0(cond_x0, uncond_x0, t, pmotion)
                else:
                    raise ValueError(f"{prediction_type} not supported!")
                pred_noise = uncond_eps + self.hparams.guidance_scale * (cond_eps - uncond_eps)
                # tmp = pmotion.max()
                if isinstance(self.sample_scheduler, UniPCMultistepScheduler) or isinstance(self.sample_scheduler,
                                                                                            DDPMScheduler):
                    pmotion = self.sample_scheduler.step(pred_noise, t, pmotion).prev_sample.float()
                else:
                    pmotion = self.sample_scheduler.step(pred_noise, t, pmotion,
                                                         use_clipped_model_output=False).prev_sample.float()
                # print(i, t, tmp, pmotion.max())

            final_motion_gen[pmap == 1] = pmotion[~pmask]

        # assert torch.sum(torch.abs(final_motion_gen[motion_prior==1] - gt_motion[motion_prior==1])) < 1e-6
        if self.hparams.is_mask_init_prior:
            final_motion_gen[motion_prior.bool()] = gt_motion[motion_prior.bool()]
        return final_motion_gen, length.clone()

    @torch.no_grad()
    def evaluate_model(self, batch, split='val'):
        motion, text, length, keyframes = batch["motion"], batch["text"], batch["length"], batch["keyframes"]
        word_embs, pos_ohot, text_lengths = batch['word_embs'], batch["pos_ohot"], batch["text_len"]

        if self.hparams.motion_repr != "hml3d":
            pmg_motion = batch["pmg_motion"]
            raw_motion = motion.clone()
            motion = pmg_motion.clone()

        if split == 'test':
            dataset = self.trainer.test_dataloaders.dataset
        else:
            dataset = self.trainer.val_dataloaders.dataset

        if split == 'test' and self.trainer.test_dataloaders.dataset.is_mm:
            motion_gen = []
            repeats = self.hparams.metrics.mm_num_repeats
            num_keyframes = self.trainer.test_dataloaders.dataset.keyframe_info.num
            # num_keyframes = 1
            type_keyframes = self.trainer.test_dataloaders.dataset.keyframe_info.type
            for _ in range(repeats):
                mm_keyframes = generate_prior_info(motion, length, "%d_%s" % (num_keyframes, type_keyframes))
                _motion_gen, _len_gen = self.sample_motion(motion, text, length, mm_keyframes)
                motion_gen.append(_motion_gen)
            motion_gen = torch.cat(motion_gen, dim=0)
            text = text * repeats
            motion = motion.repeat([repeats, 1, 1])
            raw_motion = raw_motion.repeat([repeats, 1, 1])
            length = length.repeat([repeats])
            len_gen = length
        else:
            motion_gen, len_gen = self.sample_motion(motion, text, length, keyframes)



        if self.hparams.motion_repr != "hml3d":
            joints_gen = dataset.feats2joints(motion_gen, self.hparams.motion_repr)
            joints_gt = dataset.feats2joints(raw_motion)
            motion_gt_unnorm = dataset.inv_transform(raw_motion)
            pmg_motion_gen_unnorm = dataset.inv_transform(motion_gen, self.hparams.motion_repr)
            # motion_gen_unnorm = torch.zeros_like(motion_gt_unnorm).to(motion_gt_unnorm)
            motion_gen_unnorm = motion_gt_unnorm.clone()
            for i in range(len(motion_gen_unnorm)):
                motion_gen_unnorm[i, :length[i]] = convert_motion_representation(pmg_motion_gen_unnorm[i, :length[i]],
                                                            src_repr=self.hparams.motion_repr, tgt_repr="hml3d")
        else:
            joints_gen = dataset.feats2joints(motion_gen)
            joints_gt = dataset.feats2joints(motion)
            motion_gen_unnorm = dataset.inv_transform(motion_gen)
            motion_gt_unnorm = dataset.inv_transform(motion)

        if split == 'test' and self.trainer.test_dataloaders.dataset.is_mm:
            mc_motion_gen_emb = self.motionclip_evaluator.extract_motion_embedding(motion_gen_unnorm, len_gen)
            t2m_motion_gen_emb = self.t2m_evaluator.extract_motion_embedding(motion_gen_unnorm, len_gen)

            bz = len(batch["motion"])
            self.eval_metrics["motionclip_mm_gen"].update(mc_motion_gen_emb.view([repeats, bz, -1]).permute([1, 0, 2]))
            self.eval_metrics["t2m_mm_gen"].update(t2m_motion_gen_emb.view([repeats, bz, -1]).permute([1, 0, 2]))
        else:
            # MotionCLIP
            mc_text_emb = self.motionclip_evaluator.extract_text_embedding(motion.device, text)
            mc_motion_gen_emb = self.motionclip_evaluator.extract_motion_embedding(motion_gen_unnorm, len_gen)
            mc_motion_gt_emb = self.motionclip_evaluator.extract_motion_embedding(motion_gt_unnorm, length)

            # T2MCLIP
            t2m_text_emb = self.t2m_evaluator.extract_text_embedding(word_embs, pos_ohot, text_lengths, motion.device)
            t2m_motion_gen_emb = self.t2m_evaluator.extract_motion_embedding(motion_gen_unnorm, len_gen)
            t2m_motion_gt_emb = self.t2m_evaluator.extract_motion_embedding(motion_gt_unnorm, length)

            self.eval_metrics["motionclip"].update(mc_text_emb, mc_motion_gen_emb, mc_motion_gt_emb, length)
            self.eval_metrics["t2m"].update(t2m_text_emb, t2m_motion_gen_emb, t2m_motion_gt_emb, length)
            self.eval_metrics["CoordinateError"].update(joints_gen, joints_gt, length)

            # MoBert
            if self.hparams.dataset_name == "hml3d":
                alignment, faithfulness_rating, naturalness_rating = self.mobert_evaluator.calculate_score(
                    motion_gen_unnorm, len_gen, text)
                self.eval_metrics["mobert_alignment"].update(alignment)
                self.eval_metrics["mobert_faithfulness"].update(faithfulness_rating)
                self.eval_metrics["mobert_naturalness"].update(naturalness_rating)
                alignment, faithfulness_rating, naturalness_rating = self.mobert_evaluator.calculate_score(
                    motion_gt_unnorm, len_gen, text)
                self.eval_metrics["mobert_gt_alignment"].update(alignment)
                self.eval_metrics["mobert_gt_faithfulness"].update(faithfulness_rating)
                self.eval_metrics["mobert_gt_naturalness"].update(naturalness_rating)

    def cal_loss(self, out):
        prediction_type = self.noise_scheduler.config.prediction_type
        mask, model_output = out["mask"], out["output"]
        if prediction_type == 'epsilon':
            loss_mse = (model_output - out["noise"]) ** 2
        elif prediction_type == 'sample':
            loss_mse = (model_output - out["x_0"]) ** 2
        else:
            raise ValueError(f"{prediction_type} not supported!")
        return {"loss": loss_mse[~mask].mean()}

    def obtain_eps_when_predicting_x_0(self, cond_x0, uncond_x0, timestep, x_t):
        scheduler = self.sample_scheduler
        alpha_prod_t = scheduler.alphas_cumprod[timestep]
        beta_prod_t = 1 - alpha_prod_t

        cond_eps = (x_t - alpha_prod_t ** 0.5 * cond_x0) / beta_prod_t ** 0.5
        uncond_eps = (x_t - alpha_prod_t ** 0.5 * uncond_x0) / beta_prod_t ** 0.5
        return cond_eps, uncond_eps

