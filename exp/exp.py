import torch
import numpy as np
import sys
import logging
import PIL

import dataset
import modules

sys.path.append("../")
import co
import config
import os

class Worker(co.mytorch.Worker):
    def __init__(
        self,
        train_n_nbs=1,
        train_scale=1,
        train_patch=64,
        eval_n_nbs=1,
        eval_scale=-1,
        n_train_iters=config.train_iters,  # 750000
        num_workers=2,
        **kwargs,
    ):
        super().__init__(
            n_train_iters=n_train_iters,
            num_workers=num_workers,
            use_cuda=config.use_cuda,
            **kwargs,
        )

        self.train_n_nbs = train_n_nbs
        self.train_scale = train_scale
        self.train_patch = train_patch
        self.eval_n_nbs = eval_n_nbs
        self.eval_scale = train_scale if eval_scale <= 0 else eval_scale
        self.bwd_depth_thresh = 0.01
        self.invalid_depth_to_inf = True

        self.train_loss = modules.CalculateLoss([config.lr, config.T_period_gan, config.eta_min, config.restarts_gan, config.restart_weights_gan],
                                                self.train_patch, config.gan_type, config.use_perceptual_gan_loss)
        self.l1_loss = torch.nn.L1Loss().cuda()

        self.eval_loss = self.train_loss

    def get_pw_dataset(
        self,
        *,
        name,
        ibr_dir,
        im_size,
        patch,
        pad_width,
        train,
        ref_num,
        scale,
        use_complement,
    ):
        logging.info(f"  create dataset for {name}")
        image_dir = str(ibr_dir)

        dset = dataset.Dataset(
            name=name,
            im_size=im_size,
            pad_width=pad_width,
            patch=patch,
            train=train,
            image_dir=image_dir,
            ref_num=ref_num,
            scale = scale,
            use_complement=use_complement,

        )
        return dset


    def get_train_set_tat(self, dset):
        dense_dir = config.tat_root / dset / "dense"
        ibr_dir = dense_dir / f"ibr3d_pw_{self.train_scale:.2f}"
        dset = self.get_pw_dataset(
            name=f'tat_{dset.replace("/", "_")}',
            ibr_dir=ibr_dir,
            im_size=None,
            pad_width=config.pad_width,
            patch=(self.train_patch[0], self.train_patch[1]),
            train=True,
            ref_num=config.ref_num,
            scale=config.scale,
            use_complement=config.use_complement,
        )
        return dset

    def get_train_set(self):
        logging.info("Create train datasets")
        dsets = co.mytorch.MultiDataset(name="train")
        for dset in config.tat_train_sets:
            dsets.append(self.get_train_set_tat(dset))
        return dsets

    def get_eval_set_tat(self, dset, mode):
        dense_dir = config.tat_root / dset / "dense"
        ibr_dir = dense_dir / f"ibr3d_pw_{self.eval_scale:.2f}"
        dset = self.get_pw_dataset(
            name=f'tat_{mode}_{dset.replace("/", "_")}',
            ibr_dir=ibr_dir,
            im_size=None,
            pad_width=config.pad_width,
            patch=None,
            train=False,
            ref_num=config.ref_num,
            scale=config.scale,
            use_complement=config.use_complement,
        )
        return dset

    def get_eval_sets(self):
        logging.info("Create eval datasets")
        eval_sets = []
        for dset in config.tat_eval_sets:
            dset = self.get_eval_set_tat(dset, "all")
            eval_sets.append(dset)
        for dset in eval_sets:
            dset.logging_rate = 1
            dset.vis_ind = np.arange(len(dset))
        return eval_sets

    def copy_data(self, data, use_cuda, train):
        self.data = {}
        for k, v in data.items():
            if use_cuda == True:
                v = v.cuda()
            self.data[k] = v.requires_grad_(requires_grad=False)

    def net_forward(self, net, train, iter):
        return net(**self.data)

    def loss_forward(self, output, train, iter):
        errs = {}

        est = output["out"]
        tgt = self.data["img_hr"].view(*est.shape)

        est = est[..., : tgt.shape[-2], : tgt.shape[-1]]

        output["out"] = est


        key = 'rgb'
        if train:
            hr_loss = self.train_loss(est, tgt, True)
            total_loss = hr_loss
            for lidx, loss in enumerate(total_loss):
                errs[key + f"{lidx}"] = loss / config.train_batch_size

        else:
            est = torch.clamp(est, -1, 1)
            est = 255 * (est + 1) / 2
            est = est.type(torch.uint8)
            est = est.type(torch.float32)
            est = (est / 255 * 2) - 1

            errs[key] = self.eval_loss(est, tgt, False)

        return errs

    def callback_eval_start(self, **kwargs):
        self.metric = None

    def im_to2np(self, im ):
        im = im.detach().to("cpu").numpy()
        im = (np.clip(im, -1, 1) + 1) / 2
        im = im.transpose(0, 2, 3, 1)
        return im

    def callback_eval_add(self, **kwargs):
        output = kwargs["output"]
        batch_idx = kwargs["batch_idx"]
        iter = kwargs["iter"]
        eval_set = kwargs["eval_set"]
        eval_set_name = eval_set.name.replace("/", "_")
        eval_set_name = f"{eval_set_name}_{self.eval_scale}"


        B, C, H, W = self.data["img_hr"].shape
        size=  self.data["HR_size"]
        ta = self.im_to2np(self.data["img_hr"])[:, : size[0,0], : size[0,1], :]

        out_dir = self.exp_out_root / f"{eval_set_name}_n{self.eval_n_nbs}" / f"{iter}"
        out_dir.mkdir(parents=True, exist_ok=True)


        es = self.im_to2np(output["out"])[:, : size[0,0], : size[0,1], :]

        key = 'rgb'
        vec_length = 3
        if self.metric is None:
            self.metric = {}

            self.metric[key] = co.metric.MultipleMetric(
                metrics=[
                    co.metric.DistanceMetric(p=1, vec_length=vec_length),
                ]
            )
        self.metric[key].add(es, ta)

        masks = output["ref_align_mask"]
        for mask_id, mask in enumerate(masks):
            mask1 = mask[..., : size[0,0], : size[0,1]].detach().to("cpu").numpy()
            for b in range(mask1.shape[0]):
                if not os.path.exists(str(out_dir / f"{batch_idx:04d}")):
                    os.makedirs(str(out_dir / f"{batch_idx:04d}"))
                for c in range(mask1.shape[1]):
                    out_im = (255 * mask1[b, c]).astype(np.uint8)
                    out_im = np.squeeze(out_im)
                    PIL.Image.fromarray(out_im).save(out_dir / f"{batch_idx:04d}" / f"masks_{b:04d}_{c:04d}_{mask_id:04d}_es.png")

        for b in range(ta.shape[0]):
            bidx = batch_idx
            out_im = (255 * es[b]).astype(np.uint8)
            if not os.path.exists(str(out_dir / f"{bidx:04d}")):
                os.makedirs(str(out_dir / f"{bidx:04d}"))
            PIL.Image.fromarray(out_im).save(out_dir / f"{bidx:04d}"/ f"s{b:04d}_es.png")

    def callback_eval_stop(self, **kwargs):
        eval_set = kwargs["eval_set"]
        eval_set_name = eval_set.name.replace("/", "_")

if __name__ == "__main__":
    parser = co.mytorch.get_parser()
    args = parser.parse_args()
    experiment_name = config.experiment_name

    train_scale = 0.50
    worker = Worker(
        experiment_name=experiment_name,
        train_n_nbs=config.ref_num,
        train_scale=train_scale,
        train_patch=config.train_patch,
        eval_n_nbs=config.ref_num,
        eval_scale=train_scale,
    )
    worker.save_frequency = co.mytorch.Frequency(hours=2)
    worker.eval_frequency = co.mytorch.Frequency(hours=2)
    worker.train_batch_size = config.train_batch_size
    worker.eval_batch_size = config.eval_batch_size
    worker.train_batch_acc_steps = 1

    worker_objects = co.mytorch.WorkerObjects(
        optim_f=lambda net: torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=config.lr, betas=(config.beta1, config.beta2))
    )

    worker_objects.net_f = lambda: modules.get_mvsr_net(nf=config.nf, ref_num=config.ref_num)
    worker.do(args, worker_objects)
