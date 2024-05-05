"""Pytorch script for training FoldFlow."""
import os

# This line magically changes some tensors to double precision
# so we need to reset the default dtype later.
os.environ["GEOMSTATS_BACKEND"] = "pytorch"
import copy
import logging
import time
from collections import defaultdict, deque

import GPUtil
import hydra
import numpy as np
import pandas as pd
import torch
import tree
import wandb
from einops import rearrange
import foldflow.utils.experiments_utils as eu
from hydra.core.hydra_config import HydraConfig
from lightning import Fabric
from omegaconf import DictConfig, OmegaConf
from torch.nn import DataParallel as DP
from foldflow.utils.so3_helpers import hat_inv, pt_to_identity

from foldflow.data import all_atom, pdb_data_loader
from foldflow.data import utils as du
from foldflow.models import se3_fm
from foldflow.models.components import network
from openfold.utils import rigid_utils as ru
from tools.analysis import metrics
from tools.analysis import utils as au


class Experiment:
    def __init__(
        self,
        *,
        conf: DictConfig,
    ):
        """Initialize experiment.

        Args:
            exp_cfg: Experiment configuration.
        """
        self.first_batch = None
        self._log = logging.getLogger(__name__)
        self._available_gpus = "".join(
            [str(x) for x in GPUtil.getAvailable(order="memory", limit=8)]
        )

        # Configs
        self._conf = conf
        self._exp_conf = conf.experiment
        if HydraConfig.initialized() and "num" in HydraConfig.get().job:
            self._exp_conf.name = f"{self._exp_conf.name}_{HydraConfig.get().job.num}"
        self._fm_conf = conf.flow_matcher
        self._model_conf = conf.model
        self._data_conf = conf.data
        self._wandb_conf = conf.wandb
        self._use_wandb = self._wandb_conf.use_wandb
        self._use_ddp = self._exp_conf.use_ddp
        # 1. initialize ddp info if in ddp mode
        # 2. silent rest of logger when use ddp mode
        # 3. silent wandb logger
        # 4. unset checkpoint path if rank is not 0 to avoid saving checkpoints and evaluation
        print(f"Number of threads {self._exp_conf.torch_num_threads}")
        torch.set_num_threads(self._exp_conf.torch_num_threads)
        # reduce matmul precision for better performance on GPU
        torch.set_float32_matmul_precision("medium")
        torch.set_default_dtype(torch.float32)
        torch.backends.cuda.matmul.allow_tf32 = True
        self._master_proc = True

        if self._use_ddp:
            from lightning.fabric.strategies import DDPStrategy

            strategy = DDPStrategy(find_unused_parameters=False)
            self.fabric = Fabric(
                accelerator="cuda", devices=self._exp_conf.num_gpus, strategy=strategy
            )
            self.fabric.launch()

            torch.backends.cuda.matmul.allow_tf32 = True
            self._log.info(f"Using DDP with {self.fabric.global_rank} rank")
            print(
                f"Torch uses cuddn {torch.backends.cudnn.enabled} and"
                f" cudnn benchmark {torch.backends.cudnn.benchmark}"
            )

            self.ddp_info = eu.get_ddp_info()
            self._master_proc = self.fabric.global_rank == 0
            self._global_rank = self.fabric.global_rank
            print(
                f"RANK: {self.fabric.global_rank} | master process: {self._master_proc}"
            )

            if self.fabric.global_rank != 0:
                self._log.addHandler(logging.NullHandler())
                self._log.setLevel("ERROR")
                self._use_wandb = False
                # self._exp_conf.full_ckpt_dir = None

        ckpt_model, ckpt_opt = self.handle_warmstart(conf)

        if self._use_ddp and self.fabric.global_rank != 0:
            self._exp_conf.full_ckpt_dir = None

        # Initialize experiment objects
        self._flow_matcher = se3_fm.SE3FlowMatcher(self._fm_conf)

        self._model = network.VectorFieldNetwork(self._model_conf, self.flow_matcher)
        if ckpt_model is not None:
            ckpt_model = {k.replace("module.", ""): v for k, v in ckpt_model.items()}
            ckpt_model = {
                k.replace("score_model.", "vectorfield."): v
                for k, v in ckpt_model.items()
            }
            self._model.load_state_dict(ckpt_model, strict=True)

        num_parameters = sum(p.numel() for p in self._model.parameters())
        self._exp_conf.num_parameters = num_parameters
        self._log.info(f"Number of model parameters {num_parameters}")
        self._optimizer = torch.optim.Adam(
            self._model.parameters(), lr=self._exp_conf.learning_rate
        )
        if ckpt_opt is not None:
            self._optimizer.load_state_dict(ckpt_opt)
            if conf.experiment.use_gpu:
                for state in self._optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.cuda()

        if self._exp_conf.full_ckpt_dir is not None:
            # Set-up checkpoint location
            ckpt_dir = self._exp_conf.full_ckpt_dir
            if not os.path.exists(ckpt_dir):
                os.makedirs(ckpt_dir, exist_ok=True)
            self._exp_conf.full_ckpt_dir = ckpt_dir
            self._log.info(f"Checkpoints saved to: {ckpt_dir}")
        else:
            self._log.info("Checkpoint not being saved.")
        if self._exp_conf.eval_dir is not None:
            eval_dir = os.path.join(
                self._exp_conf.eval_dir,
                self._exp_conf.name,
                self.conf.start_time_string,
            )
            self._exp_conf.eval_dir = eval_dir
            self._log.info(f"Evaluation saved to: {eval_dir}")
        else:
            self._exp_conf.eval_dir = os.devnull
            self._log.info("Evaluation will not be saved.")
        self._aux_data_history = deque(maxlen=100)

        # DEBUG Variables
        self._first_train_feats = None
        self._global_rank = 0

    def handle_warmstart(self, conf):
        # Warm starting
        ckpt_model = None
        ckpt_opt = None
        self.trained_epochs = 0
        self.trained_steps = 0
        if not conf.experiment.warm_start:
            return None, None

        assert conf.experiment.warm_start in ["auto", "force"]

        # check path exists
        full_ckpt_dir = conf.experiment.full_ckpt_dir
        print(f"THIS IS TH FULL CONF!\n {conf.experiment}")
        if full_ckpt_dir is not None and not os.path.exists(full_ckpt_dir):
            if conf.experiment.warm_start == "auto":
                return None, None
            if conf.experiment.warm_start == "force":
                raise ValueError(f"full_ckpt_dir {full_ckpt_dir} does not exist")

        ckpt_files = [x for x in os.listdir(full_ckpt_dir) if "pkl" in x or ".pth" in x]
        if len(ckpt_files) == 0:
            if conf.experiment.warm_start == "auto":
                return None, None
            if conf.experiment.warm_start == "force":
                raise ValueError(f"full_ckpt_dir {full_ckpt_dir} has no checkpoints")

        self._log.info(f"Warm starting from: {full_ckpt_dir}")

        ckpt_name = ckpt_files[0]
        if len(ckpt_files) != 1:
            paths = [os.path.join(full_ckpt_dir, ckpt_file) for ckpt_file in ckpt_files]
            ckpt_name = max(paths, key=os.path.getmtime).split("/")[-1]
            self._log.info("Loading most recent ckpt")
        ckpt_path = os.path.join(full_ckpt_dir, ckpt_name)
        self._log.info(f"Loading checkpoint from {ckpt_path}")
        ckpt_pkl = du.read_pkl(ckpt_path, use_torch=True)
        ckpt_model = ckpt_pkl["model"]

        if conf.experiment.use_warm_start_conf:
            OmegaConf.set_struct(conf, False)
            conf = OmegaConf.merge(conf, ckpt_pkl["conf"])
            OmegaConf.set_struct(conf, True)
        conf.experiment.warm_start = full_ckpt_dir

        # For compatibility with older checkpoints.
        if "optimizer" in ckpt_pkl:
            ckpt_opt = ckpt_pkl["optimizer"]
        if "epoch" in ckpt_pkl:
            self.trained_epochs = ckpt_pkl["epoch"]
        if "step" in ckpt_pkl:
            self.trained_steps = ckpt_pkl["step"]
        return ckpt_model, ckpt_opt

    @property
    def flow_matcher(self):
        return self._flow_matcher

    @property
    def model(self):
        return self._model

    @property
    def conf(self):
        return self._conf

    def create_dataset(self):
        train_dataset = pdb_data_loader.PdbDataset(
            data_conf=self._data_conf,
            gen_model=self._flow_matcher,
            is_training=True,
            is_OT=self._fm_conf.ot_plan,
            ot_fn=self._fm_conf.ot_fn,
            reg=self._fm_conf.reg,
        )

        valid_dataset = pdb_data_loader.PdbDataset(
            data_conf=self._data_conf,
            gen_model=self._flow_matcher,
            is_OT=self._fm_conf.ot_plan,
            ot_fn=self._fm_conf.ot_fn,
            reg=self._fm_conf.reg,
            is_training=False,
        )
        if self._use_ddp:
            train_sampler = pdb_data_loader.DistributedTrainSampler(
                data_conf=self._data_conf,
                dataset=train_dataset,
                batch_size=self._exp_conf.batch_size,
                sample_mode=self._exp_conf.sample_mode,
                rank=self.fabric.global_rank,
                max_squared_res=self._exp_conf.max_squared_res,
                num_gpus=self._exp_conf.num_gpus,  # TODO fix arg based on actual fabric
            )
        else:
            train_sampler = pdb_data_loader.TrainSampler(
                data_conf=self._data_conf,
                dataset=train_dataset,
                batch_size=self._exp_conf.batch_size,
                sample_mode=self._exp_conf.sample_mode,
                max_squared_res=self._exp_conf.max_squared_res,
                num_gpus=self._exp_conf.num_gpus,
            )
        valid_sampler = None
        num_workers = self._exp_conf.num_loader_workers

        train_loader = du.create_data_loader(
            train_dataset,
            sampler=train_sampler,
            np_collate=False,
            length_batch=True,
            batch_size=self._exp_conf.batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
            max_squared_res=self._exp_conf.max_squared_res,
        )

        valid_loader = du.create_data_loader(
            valid_dataset,
            sampler=valid_sampler,
            np_collate=False,
            length_batch=True,
            batch_size=self._exp_conf.eval_batch_size,
            shuffle=False,
            num_workers=0,
            drop_last=False,
        )

        if self._exp_conf.use_ddp:
            train_loader = self.fabric.setup_dataloaders(
                train_loader, use_distributed_sampler=False
            )
        return train_loader, valid_loader, train_sampler, valid_sampler

    def init_wandb(self):
        self._log.info("Initializing Wandb.")
        conf_dict = OmegaConf.to_container(self._conf, resolve=True)
        if self._exp_conf.run_id is None:
            self._exp_conf.run_id = wandb.util.generate_id()
        wandb.init(
            project=self._wandb_conf.project,
            entity=self._wandb_conf.entity,
            name=self._exp_conf.name,
            config=dict(eu.flatten_dict(conf_dict)),
            dir=self._wandb_conf.dir,
            id=self._exp_conf.run_id,
            tags=self._wandb_conf.tags,
            group=self._wandb_conf.group,
            mode="offline" if self._wandb_conf.offline else "online",
            job_type=self._wandb_conf.job_type,
            resume="auto" if self._exp_conf.warm_start is not None else None,
        )
        self._wandb_conf.dir = wandb.run.dir
        self._log.info(
            f"Wandb: run_id={self._exp_conf.run_id}, run_dir={self._wandb_conf.dir}"
        )

    def start_training(self, return_logs=False):
        # print(f"Start-training-"*10)
        # Set environment variables for which GPUs to use.
        if HydraConfig.initialized() and "num" in HydraConfig.get().job:
            replica_id = int(HydraConfig.get().job.num)
        else:
            replica_id = 0
        if self._use_wandb and replica_id == 0:
            self.init_wandb()
        assert not self._exp_conf.use_ddp or self._exp_conf.use_gpu

        # GPU mode
        if torch.cuda.is_available() and self._exp_conf.use_gpu:
            # single GPU mode
            if self._exp_conf.num_gpus == 1:
                try:
                    gpu_id = self._available_gpus[replica_id]
                    device = f"cuda:{gpu_id}"
                except IndexError:
                    device = "cuda:0"
                    self._log.warning("Error on available gpus, trying with device 0")
                self._model = self.model.to(device)
                self._log.info(f"Using device: {device}")
            # muti gpu mode
            elif self._exp_conf.num_gpus > 1:
                # DDP mode
                if self._use_ddp:
                    self._model, self._optimizer = self.fabric.setup(
                        self._model, self._optimizer
                    )
                    device = self.fabric.device
                    self._log.info(f"Using device: {device}")
                # DP mode
                else:
                    device_ids = [
                        f"cuda:{i}"
                        for i in self._available_gpus[: self._exp_conf.num_gpus]
                    ]
                    if len(self._available_gpus) > self._exp_conf.num_gpus:
                        raise ValueError(
                            f"require {self._exp_conf.num_gpus} GPUs, but only {len(self._available_gpus)} GPUs available "
                        )
                    self._log.info(
                        f"Multi-GPU training on GPUs in DP mode: {device_ids}"
                    )
                    gpu_id = self._available_gpus[replica_id]
                    device = f"cuda:{gpu_id}"
                    self._model = DP(self._model, device_ids=device_ids)
                    self._model = self.model.to(device)
        else:
            device = "cpu"
            self._log.info(f"Using device: {device}")
            self._model = self.model.to(device)

        self._model.train()
        (
            train_loader,
            valid_loader,
            train_sampler,
            valid_sampler,
        ) = self.create_dataset()

        logs = []
        for epoch in range(self.trained_epochs, self._exp_conf.num_epoch):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            if valid_sampler is not None:
                valid_sampler.set_epoch(epoch)
            self.trained_epochs = epoch
            epoch_log = self.train_epoch(
                train_loader, valid_loader, device, return_logs=return_logs
            )
            if return_logs:
                logs.append(epoch_log)

        self._log.info("Done")
        if return_logs:
            return logs
        return 0

    def update_fn(self, data, debug=False):
        """Updates the state using some data and returns metrics."""
        self._optimizer.zero_grad()
        # torch.autograd.set_detect_anomaly(True, check_nan=True)

        loss, aux_data = self.loss_fn(data)
        if self._use_ddp:
            self.fabric.backward(loss)
        else:
            loss.backward()

        if debug:
            for name, param in self._model.named_parameters():
                if param.grad is None:
                    print(f"NO GRAD FOR PARAMETERS  {name}")

        torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
        self._optimizer.step()
        return loss, aux_data

    def train_epoch(self, train_loader, valid_loader, device, return_logs=False):
        log_lossses = defaultdict(list)
        global_logs = []
        log_time = time.time()
        step_time = time.time()

        for train_feats in train_loader:
            if "dummy_batch" in train_feats:
                self._log.error("Dummy batch")
                continue

            if not self._use_ddp:
                train_feats = tree.map_structure(lambda x: x.to(device), train_feats)

            loss, aux_data = self.update_fn(train_feats)

            if return_logs:
                global_logs.append(loss)
            for k, v in aux_data.items():
                log_lossses[k].append(du.move_to_np(v))
            self.trained_steps += 1

            # Logging to terminal
            if (
                self.trained_steps == 1
                or self.trained_steps % self._exp_conf.log_freq == 0
            ):
                elapsed_time = time.time() - log_time
                log_time = time.time()
                step_per_sec = self._exp_conf.log_freq / elapsed_time
                rolling_losses = tree.map_structure(np.mean, log_lossses)
                loss_log = " ".join(
                    [
                        f"{k}={v[0]:.4f}"
                        for k, v in rolling_losses.items()
                        if "batch" not in k
                    ]
                )
                self._log.info(
                    f"[{self.trained_steps}]: {loss_log}, steps/sec={step_per_sec:.5f}"
                )
                log_lossses = defaultdict(list)
            # Take checkpoint
            if ((self.trained_steps % self._exp_conf.ckpt_freq) == 0) or (
                self._exp_conf.early_ckpt and self.trained_steps == 10
            ):
                if self._master_proc and self._exp_conf.full_ckpt_dir is not None:
                    self._log.info("Take checkpoint")
                    ckpt_path = os.path.join(
                        self._exp_conf.full_ckpt_dir, f"step_{self.trained_steps}.pth"
                    )
                    du.write_checkpoint(
                        ckpt_path,
                        self.model.state_dict(),
                        self._conf,
                        self._optimizer.state_dict(),
                        self.trained_epochs,
                        self.trained_steps,
                        logger=self._log,
                        use_torch=True,
                    )
            ckpt_metrics = None
            eval_time = None
            if ((self.trained_steps % self._exp_conf.eval_freq) == 0) or (
                self._exp_conf.early_ckpt and self.trained_steps == 10
            ):
                if self._master_proc:
                    # Run evaluation
                    start_time = time.time()
                    eval_dir = os.path.join(
                        self._exp_conf.eval_dir, f"step_{self.trained_steps}"
                    )
                    os.makedirs(eval_dir, exist_ok=True)

                    self._log.info(
                        f"Running evaluation at EP "
                        f"{self.trained_epochs} step {self.trained_steps} in {eval_dir}"
                    )

                    ckpt_metrics = self.eval_fn(
                        eval_dir,
                        valid_loader,
                        device,
                        noise_scale=self._exp_conf.noise_scale,
                    )
                    eval_time = time.time() - start_time
                    self._log.info(f"Finished evaluation in {eval_time:.2f}s")

            # Remote log to Wandb.
            if self._use_wandb and self._master_proc:
                step_time = time.time() - step_time
                example_per_sec = self._exp_conf.batch_size / step_time
                step_time = time.time()
                wandb_logs = {
                    "loss": loss,
                    "rotation_loss": aux_data["rot_loss"],
                    "translation_loss": aux_data["trans_loss"],
                    "bb_atom_loss": aux_data["bb_atom_loss"],
                    "dist_mat_loss": aux_data["batch_dist_mat_loss"],
                    "batch_size": aux_data["examples_per_step"],
                    "res_length": aux_data["res_length"],
                    "examples_per_sec": example_per_sec,
                    "num_epochs": self.trained_epochs,
                }

                # Stratified losses
                wandb_logs.update(
                    eu.t_stratified_loss(
                        du.move_to_np(train_feats["t"]),
                        du.move_to_np(aux_data["batch_rot_loss"]),
                        loss_name="rot_loss",
                    )
                )

                wandb_logs.update(
                    eu.t_stratified_loss(
                        du.move_to_np(train_feats["t"]),
                        du.move_to_np(aux_data["batch_trans_loss"]),
                        loss_name="trans_loss",
                    )
                )

                wandb_logs.update(
                    eu.t_stratified_loss(
                        du.move_to_np(train_feats["t"]),
                        du.move_to_np(aux_data["batch_bb_atom_loss"]),
                        loss_name="bb_atom_loss",
                    )
                )

                wandb_logs.update(
                    eu.t_stratified_loss(
                        du.move_to_np(train_feats["t"]),
                        du.move_to_np(aux_data["batch_dist_mat_loss"]),
                        loss_name="dist_mat_loss",
                    )
                )

                if ckpt_metrics is not None:
                    wandb_logs["eval_time"] = eval_time
                    for metric_name in metrics.ALL_METRICS:
                        wandb_logs[metric_name] = ckpt_metrics[metric_name].mean()
                    eval_table = wandb.Table(
                        columns=ckpt_metrics.columns.to_list() + ["structure"]
                    )
                    for _, row in ckpt_metrics.iterrows():
                        pdb_path = row["sample_path"]
                        row_metrics = row.to_list() + [wandb.Molecule(pdb_path)]
                        eval_table.add_data(*row_metrics)
                    wandb_logs["sample_metrics"] = eval_table

                wandb.log(wandb_logs, step=self.trained_steps)

            if torch.isnan(loss):
                if self._use_wandb:
                    wandb.alert(
                        title="Encountered NaN loss",
                        text=f"Loss NaN after {self.trained_epochs} epochs, {self.trained_steps} steps",
                    )
                raise Exception("NaN encountered")

        if return_logs:
            return global_logs

    def eval_fn(
        self,
        eval_dir,
        valid_loader,
        device,
        min_t=None,
        num_t=None,
        noise_scale=1.0,
        context=None,
    ):
        ckpt_eval_metrics = []
        for valid_feats, pdb_names in valid_loader:
            res_mask = du.move_to_np(valid_feats["res_mask"].bool())
            fixed_mask = du.move_to_np(valid_feats["fixed_mask"].bool())
            aatype = du.move_to_np(valid_feats["aatype"])
            gt_prot = du.move_to_np(valid_feats["atom37_pos"])
            batch_size = res_mask.shape[0]
            valid_feats = tree.map_structure(lambda x: x.to(device), valid_feats)

            # Run inference
            infer_out = self.inference_fn(
                valid_feats,
                min_t=min_t,
                num_t=num_t,
                noise_scale=noise_scale,
                context=context,
            )
            final_prot = infer_out["prot_traj"][0]
            for i in range(batch_size):
                num_res = int(np.sum(res_mask[i]).item())
                unpad_fixed_mask = fixed_mask[i][res_mask[i]]
                unpad_flow_mask = 1 - unpad_fixed_mask
                unpad_prot = final_prot[i][res_mask[i]]
                unpad_gt_prot = gt_prot[i][res_mask[i]]
                unpad_gt_aatype = aatype[i][res_mask[i]]
                percent_flowed = np.sum(unpad_flow_mask) / num_res
                if self._use_ddp:
                    prot_path = os.path.join(
                        eval_dir,
                        f"len_{num_res}_sample_{i}_{self.fabric.device}_flowed_{percent_flowed:.2f}.pdb",
                    )
                else:
                    prot_path = os.path.join(
                        eval_dir,
                        f"len_{num_res}_sample_{i}_flowed_{percent_flowed:.2f}.pdb",
                    )

                # Extract argmax predicted aatype
                saved_path = au.write_prot_to_pdb(
                    unpad_prot,
                    prot_path,
                    no_indexing=True,
                    b_factors=np.tile(1 - unpad_fixed_mask[..., None], 37) * 100,
                )
                try:
                    sample_metrics = metrics.protein_metrics(
                        pdb_path=saved_path,
                        atom37_pos=unpad_prot,
                        gt_atom37_pos=unpad_gt_prot,
                        gt_aatype=unpad_gt_aatype,
                        flow_mask=unpad_flow_mask,
                    )
                except ValueError as e:
                    self._log.warning(
                        f"Failed evaluation of length {num_res} sample {i}: {e}"
                    )
                    continue
                sample_metrics["step"] = self.trained_steps
                sample_metrics["num_res"] = num_res
                sample_metrics["fixed_residues"] = np.sum(unpad_fixed_mask)
                sample_metrics["flowed_percentage"] = percent_flowed
                sample_metrics["sample_path"] = saved_path
                sample_metrics["gt_pdb"] = pdb_names[i]
                ckpt_eval_metrics.append(sample_metrics)

        # Save metrics as CSV.
        eval_metrics_csv_path = os.path.join(eval_dir, "metrics.csv")
        ckpt_eval_metrics = pd.DataFrame(ckpt_eval_metrics)
        ckpt_eval_metrics.to_csv(eval_metrics_csv_path, index=False)
        return ckpt_eval_metrics

    def _self_conditioning(self, batch):
        model_sc = self.model(batch)
        batch["sc_ca_t"] = model_sc["rigids"][..., 4:]
        return batch

    def loss_fn(self, batch):
        """Computes loss and auxiliary data.

        Args:
            batch: Batched data.
            model_out: Output of model ran on batch.

        Returns:
            loss: Final training loss scalar.
            aux_data: Additional logging data.
        """
        if (
            self._model_conf.embed.embed_self_conditioning
            and self.trained_steps % 2 == 1
        ):
            # if self._model_conf.embed.embed_self_conditioning and random.random() > 0.5:
            with torch.no_grad():
                batch = self._self_conditioning(batch)

        _, gt_rot_u_t = self._flow_matcher._so3_fm.vectorfield(
            batch["rot_vectorfield"], batch["rot_t"], batch["t"]
        )

        model_out = self.model(batch)
        bb_mask = batch["res_mask"]
        flow_mask = 1 - batch["fixed_mask"]
        loss_mask = bb_mask * flow_mask
        batch_size, num_res = bb_mask.shape

        gt_trans_u_t = batch["trans_vectorfield"]
        rot_vectorfield_scaling = batch["rot_vectorfield_scaling"]
        trans_vectorfield_scaling = batch["trans_vectorfield_scaling"]
        batch_loss_mask = torch.any(bb_mask, dim=-1)

        pred_rot_v_t = model_out["rot_vectorfield"] * flow_mask[..., None, None]
        pred_trans_v_t = model_out["trans_vectorfield"] * flow_mask[..., None]

        # Translation vectorfield loss
        trans_vectorfield_mse = (gt_trans_u_t - pred_trans_v_t) ** 2 * loss_mask[
            ..., None
        ]
        trans_vectorfield_loss = torch.sum(
            trans_vectorfield_mse / trans_vectorfield_scaling[:, None, None] ** 2,
            dim=(-1, -2),
        ) / (loss_mask.sum(dim=-1) + 1e-10)

        # Translation x0 loss
        gt_trans_x0 = batch["rigids_0"][..., 4:] * self._exp_conf.coordinate_scaling
        pred_trans_x0 = model_out["rigids"][..., 4:] * self._exp_conf.coordinate_scaling
        trans_x0_loss = torch.sum(
            (gt_trans_x0 - pred_trans_x0) ** 2 * loss_mask[..., None], dim=(-1, -2)
        ) / (loss_mask.sum(dim=-1) + 1e-10)

        trans_loss = trans_vectorfield_loss * (
            batch["t"] > self._exp_conf.trans_x0_threshold
        ) + trans_x0_loss * (batch["t"] <= self._exp_conf.trans_x0_threshold)
        trans_loss *= self._exp_conf.trans_loss_weight
        trans_loss *= int(self._fm_conf.flow_trans)

        # Rotation loss
        # gt_rot_u_t and pred_rot_v_t are matrices convert
        t_shape = batch["rot_t"].shape[0]
        rot_t = rearrange(batch["rot_t"], "t n c d -> (t n) c d", c=3, d=3).double()
        gt_rot_u_t = rearrange(gt_rot_u_t, "t n c d -> (t n) c d", c=3, d=3)
        pred_rot_v_t = rearrange(pred_rot_v_t, "t n c d -> (t n) c d", c=3, d=3)
        try:
            rot_t = rot_t.double()
            gt_at_id = pt_to_identity(rot_t, gt_rot_u_t)
            gt_rot_u_t = hat_inv(gt_at_id)
            pred_at_id = pt_to_identity(rot_t, pred_rot_v_t)
            pred_rot_v_t = hat_inv(pred_at_id)
        except ValueError as e:
            self._log.info(
                f"Skew symmetric error gt {((gt_at_id + gt_at_id.transpose(-1, -2))**2).mean()} "
                f"pred {((pred_at_id + pred_at_id.transpose(-1, -2))**2).mean()} Skipping rot loss"
            )
            gt_rot_u_t = torch.zeros_like(rot_t[..., 0])
            pred_rot_v_t = torch.zeros_like(rot_t[..., 0])

        gt_rot_u_t = rearrange(gt_rot_u_t, "(t n) c -> t n c", t=t_shape, c=3)
        pred_rot_v_t = rearrange(pred_rot_v_t, "(t n) c -> t n c", t=t_shape, c=3)

        if self._exp_conf.separate_rot_loss:
            gt_rot_angle = torch.norm(gt_rot_u_t, dim=-1, keepdim=True)
            gt_rot_axis = gt_rot_u_t / (gt_rot_angle + 1e-6)

            pred_rot_angle = torch.norm(pred_rot_v_t, dim=-1, keepdim=True)
            pred_rot_axis = pred_rot_v_t / (pred_rot_angle + 1e-6)

            # Separate loss on the axis
            axis_loss = (gt_rot_axis - pred_rot_axis) ** 2 * loss_mask[..., None]
            axis_loss = torch.sum(axis_loss, dim=(-1, -2)) / (
                loss_mask.sum(dim=-1) + 1e-10
            )

            # Separate loss on the angle
            angle_loss = (gt_rot_angle - pred_rot_angle) ** 2 * loss_mask[..., None]
            angle_loss = torch.sum(
                angle_loss / rot_vectorfield_scaling[:, None, None] ** 2, dim=(-1, -2)
            ) / (loss_mask.sum(dim=-1) + 1e-10)
            angle_loss *= self._exp_conf.rot_loss_weight
            angle_loss *= batch["t"] > self._exp_conf.rot_loss_t_threshold
            rot_loss = angle_loss + axis_loss
        else:
            rot_mse = (gt_rot_u_t - pred_rot_v_t) ** 2 * loss_mask[..., None]
            rot_loss = torch.sum(
                rot_mse / rot_vectorfield_scaling[:, None, None] ** 2,
                dim=(-1, -2),
            ) / (loss_mask.sum(dim=-1) + 1e-10)
            rot_loss *= self._exp_conf.rot_loss_weight
            rot_loss *= batch["t"] > self._exp_conf.rot_loss_t_threshold
        rot_loss *= int(self._fm_conf.flow_rot)

        # Backbone atom loss
        pred_atom37 = model_out["atom37"][:, :, :5]
        gt_rigids = ru.Rigid.from_tensor_7(batch["rigids_0"].type(torch.float32))
        gt_psi = batch["torsion_angles_sin_cos"][..., 2, :]
        gt_atom37, atom37_mask, _, _ = all_atom.compute_backbone(gt_rigids, gt_psi)
        gt_atom37 = gt_atom37[:, :, :5]
        atom37_mask = atom37_mask[:, :, :5]

        gt_atom37 = gt_atom37.to(pred_atom37.device)
        atom37_mask = atom37_mask.to(pred_atom37.device)
        bb_atom_loss_mask = atom37_mask * loss_mask[..., None]
        bb_atom_loss = torch.sum(
            (pred_atom37 - gt_atom37) ** 2 * bb_atom_loss_mask[..., None],
            dim=(-1, -2, -3),
        ) / (bb_atom_loss_mask.sum(dim=(-1, -2)) + 1e-10)
        bb_atom_loss *= self._exp_conf.bb_atom_loss_weight
        bb_atom_loss *= batch["t"] < self._exp_conf.bb_atom_loss_t_filter
        bb_atom_loss *= self._exp_conf.aux_loss_weight

        # Pairwise distance loss
        gt_flat_atoms = gt_atom37.reshape([batch_size, num_res * 5, 3])
        gt_pair_dists = torch.linalg.norm(
            gt_flat_atoms[:, :, None, :] - gt_flat_atoms[:, None, :, :], dim=-1
        )
        pred_flat_atoms = pred_atom37.reshape([batch_size, num_res * 5, 3])
        pred_pair_dists = torch.linalg.norm(
            pred_flat_atoms[:, :, None, :] - pred_flat_atoms[:, None, :, :], dim=-1
        )

        flat_loss_mask = torch.tile(loss_mask[:, :, None], (1, 1, 5))
        flat_loss_mask = flat_loss_mask.reshape([batch_size, num_res * 5])
        flat_res_mask = torch.tile(bb_mask[:, :, None], (1, 1, 5))
        flat_res_mask = flat_res_mask.reshape([batch_size, num_res * 5])

        gt_pair_dists = gt_pair_dists * flat_loss_mask[..., None]
        pred_pair_dists = pred_pair_dists * flat_loss_mask[..., None]
        pair_dist_mask = flat_loss_mask[..., None] * flat_res_mask[:, None, :]

        # No loss on anything >6A
        proximity_mask = gt_pair_dists < 6
        pair_dist_mask = pair_dist_mask * proximity_mask

        dist_mat_loss = torch.sum(
            (gt_pair_dists - pred_pair_dists) ** 2 * pair_dist_mask, dim=(1, 2)
        )
        dist_mat_loss /= torch.sum(pair_dist_mask, dim=(1, 2)) - num_res
        dist_mat_loss *= self._exp_conf.dist_mat_loss_weight
        dist_mat_loss *= batch["t"] < self._exp_conf.dist_mat_loss_t_filter
        dist_mat_loss *= self._exp_conf.aux_loss_weight

        final_loss = rot_loss + trans_loss + bb_atom_loss + dist_mat_loss

        def normalize_loss(x):
            return x.sum() / (batch_loss_mask.sum() + 1e-10)

        aux_data = {
            "batch_train_loss": final_loss,
            "batch_rot_loss": rot_loss,
            "batch_trans_loss": trans_loss,
            "batch_bb_atom_loss": bb_atom_loss,
            "batch_dist_mat_loss": dist_mat_loss,
            "total_loss": normalize_loss(final_loss),
            "rot_loss": normalize_loss(rot_loss),
            "trans_loss": normalize_loss(trans_loss),
            "bb_atom_loss": normalize_loss(bb_atom_loss),
            "dist_mat_loss": normalize_loss(dist_mat_loss),
            "examples_per_step": torch.tensor(batch_size),
            "res_length": torch.mean(torch.sum(bb_mask, dim=-1)),
        }

        # Maintain a history of the past N number of steps.
        # Helpful for debugging.
        self._aux_data_history.append(
            {"aux_data": aux_data, "model_out": model_out, "batch": batch}
        )

        assert final_loss.shape == (batch_size,)
        assert batch_loss_mask.shape == (batch_size,)
        return normalize_loss(final_loss), aux_data

    def _calc_trans_0(self, trans_vectorfield, trans_t, t):
        beta_t = self._flow_matcher._se3_fm._r3_fm.marginal_b_t(t)
        beta_t = beta_t[..., None, None]
        cond_var = 1 - torch.exp(-beta_t)
        return (trans_vectorfield * cond_var + trans_t) / torch.exp(-1 / 2 * beta_t)

    def _set_t_feats(self, feats, t, t_placeholder):
        feats["t"] = t * t_placeholder
        (
            rot_vectorfield_scaling,
            trans_vectorfield_scaling,
        ) = self.flow_matcher.vectorfield_scaling(t)
        feats["rot_vectorfield_scaling"] = rot_vectorfield_scaling * t_placeholder
        feats["trans_vectorfield_scaling"] = trans_vectorfield_scaling * t_placeholder
        return feats

    def forward_traj(self, x_0, min_t, num_t):
        forward_steps = np.linspace(min_t, 1.0, num_t)[:-1]
        x_traj = [x_0]
        for t in forward_steps:
            x_t = self.flow_matcher.se3_fm._r3_fm.forward(x_traj[-1], t, num_t)
            x_traj.append(x_t)
        x_traj = torch.stack(x_traj, axis=0)
        return x_traj

    def inference_fn(
        self,
        data_init,
        num_t=None,
        min_t=None,
        center=True,
        aux_traj=False,
        self_condition=True,
        noise_scale=1.0,
        context=None,
    ):
        """Inference function.

        Args:
            data_init: Initial data values for sampling.
        """

        # Run reverse process.
        sample_feats = copy.deepcopy(data_init)
        device = sample_feats["rigids_t"].device
        if sample_feats["rigids_t"].ndim == 2:
            t_placeholder = torch.ones((1,)).to(device)
        else:
            t_placeholder = torch.ones((sample_feats["rigids_t"].shape[0],)).to(device)
        if num_t is None:
            num_t = self._data_conf.num_t
        if min_t is None:
            min_t = self._data_conf.min_t
        reverse_steps = np.linspace(min_t, 1.0, num_t)[::-1]
        dt = reverse_steps[0] - reverse_steps[1]
        # dt = 1/num_t
        all_rigids = [du.move_to_np(copy.deepcopy(sample_feats["rigids_t"]))]
        all_bb_prots = []
        all_trans_0_pred = []
        all_bb_0_pred = []
        with torch.no_grad():
            if self._model_conf.embed.embed_self_conditioning and self_condition:
                sample_feats = self._set_t_feats(
                    sample_feats, reverse_steps[0], t_placeholder
                )
                sample_feats = self._self_conditioning(sample_feats)
            for t in reverse_steps:

                sample_feats = self._set_t_feats(sample_feats, t, t_placeholder)
                model_out = self.model(sample_feats)
                rot_vectorfield = model_out["rot_vectorfield"]
                trans_vectorfield = model_out["trans_vectorfield"]
                rigid_pred = model_out["rigids"]
                if self._model_conf.embed.embed_self_conditioning:
                    sample_feats["sc_ca_t"] = rigid_pred[..., 4:]
                fixed_mask = sample_feats["fixed_mask"] * sample_feats["res_mask"]
                flow_mask = (1 - sample_feats["fixed_mask"]) * sample_feats["res_mask"]
                rots_t, trans_t, rigids_t = self.flow_matcher.reverse(
                    rigid_t=ru.Rigid.from_tensor_7(sample_feats["rigids_t"]),
                    rot_vectorfield=du.move_to_np(rot_vectorfield),
                    trans_vectorfield=du.move_to_np(trans_vectorfield),
                    flow_mask=du.move_to_np(flow_mask),
                    t=t,
                    dt=dt,
                    center=center,
                    noise_scale=noise_scale,
                )

                sample_feats["rigids_t"] = rigids_t.to_tensor_7().to(device)
                if aux_traj:
                    all_rigids.append(du.move_to_np(rigids_t.to_tensor_7()))

                # Calculate x0 prediction derived from vectorfield predictions.
                gt_trans_0 = sample_feats["rigids_t"][..., 4:]
                pred_trans_0 = rigid_pred[..., 4:]
                trans_pred_0 = (
                    flow_mask[..., None] * pred_trans_0
                    + fixed_mask[..., None] * gt_trans_0
                )
                psi_pred = model_out["psi"]
                if aux_traj:
                    atom37_0 = all_atom.compute_backbone(
                        ru.Rigid.from_tensor_7(rigid_pred), psi_pred
                    )[0]
                    all_bb_0_pred.append(du.move_to_np(atom37_0))
                    all_trans_0_pred.append(du.move_to_np(trans_pred_0))
                atom37_t = all_atom.compute_backbone(rigids_t, psi_pred)[0]
                all_bb_prots.append(du.move_to_np(atom37_t))

        # Flip trajectory so that it starts from t=0.
        # This helps visualization.
        flip = lambda x: np.flip(np.stack(x), (0,))
        all_bb_prots = flip(all_bb_prots)
        if aux_traj:
            all_rigids = flip(all_rigids)
            all_trans_0_pred = flip(all_trans_0_pred)
            all_bb_0_pred = flip(all_bb_0_pred)

        ret = {
            "prot_traj": all_bb_prots,
        }
        if aux_traj:
            ret["rigid_traj"] = all_rigids
            ret["trans_traj"] = all_trans_0_pred
            ret["psi_pred"] = psi_pred[None]
            ret["rigid_0_traj"] = all_bb_0_pred
        return ret


@hydra.main(version_base=None, config_path="config/", config_name="base")
def run(conf: DictConfig) -> None:

    # Fixes bug in https://github.com/wandb/wandb/issues/1525
    os.environ["WANDB_START_METHOD"] = "thread"
    exp = Experiment(conf=conf)
    return exp.start_training()


if __name__ == "__main__":
    run()
