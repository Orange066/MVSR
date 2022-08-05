import torch.optim as optim
import config
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torchvision
import sys
import math

sys.path.append("../")
import co.lr_scheduler as lr_scheduler

def conv3x3(in_channels, out_channels, stride=1, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=True)

def srntt_init_weights(net, init_type='normal', init_gain=0.02):

    def init_func(m):
        name = m.__class__.__name__
        if hasattr(m, 'weight') and ('Conv' in name or 'Linear' in name):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError(
                    f'initialization method [{init_type}] is not implemented')
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif 'BatchNorm2d' in name:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    net.apply(init_func)

class Discriminator(nn.Module):
    def __init__(self, in_size=160, gan_type='WGAN_GP'):
        super(Discriminator, self).__init__()

        def conv_block(in_channels, out_channels):
            block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, 1, 1),
                nn.BatchNorm2d(out_channels), nn.LeakyReLU(0.2, True),
                nn.Conv2d(out_channels, out_channels, 3, 2, 1),
                nn.BatchNorm2d(out_channels), nn.LeakyReLU(0.2, True))
            return block

        ndf = 32
        in_nc = 3
        self.conv_block1 = conv_block(in_nc, ndf)
        self.conv_block2 = conv_block(ndf, ndf * 2)
        self.conv_block3 = conv_block(ndf * 2, ndf * 4)
        self.conv_block4 = conv_block(ndf * 4, ndf * 8)
        self.conv_block5 = conv_block(ndf * 8, ndf * 16)

        self.out_block = nn.Sequential(
                nn.AdaptiveAvgPool2d(1), nn.Conv2d(ndf * 16, 1024, kernel_size=1),
            nn.LeakyReLU(0.2), nn.Conv2d(1024, 1, kernel_size=1))

        srntt_init_weights(self, init_type='normal', init_gain=0.02)

    def forward(self, x):
        fea = self.conv_block1(x)
        fea = self.conv_block2(fea)
        fea = self.conv_block3(fea)
        fea = self.conv_block4(fea)
        fea = self.conv_block5(fea)

        out = self.out_block(fea)

        return out

class AdversarialLoss(nn.Module):
    def __init__(self, lr_config, use_cpu=False, num_gpu=1, gan_type='WGAN_GP', gan_k=1, train_crop_size=40):
        lr, T_period_gan, eta_min, restarts_gan, restart_weights_gan = lr_config
        super(AdversarialLoss, self).__init__()
        self.gan_type = gan_type
        self.gan_k = gan_k
        self.device = torch.device('cpu' if use_cpu else 'cuda')
        self.discriminator = Discriminator(
            train_crop_size * 4, gan_type=gan_type).cuda()
        if (num_gpu > 1):
            self.discriminator = nn.DataParallel(
                self.discriminator, list(range(num_gpu)))
        if (gan_type in ['WGAN_GP', 'GAN', 'LSGAN', 'FM']):
            self.optimizer = optim.Adam(
                self.discriminator.parameters(),
                betas=(0.9, 0.999), eps=1e-8, lr=lr
            )
            self.lr_scheduler_f = lr_scheduler.CosineAnnealingLR_Restart(
                self.optimizer, T_period_gan, eta_min=eta_min,
                restarts=restarts_gan, weights=restart_weights_gan)
        else:
            raise SystemExit('Error: no such type of GAN!')

        self.bce_loss = torch.nn.BCELoss().cuda()
        self.mse_loss = torch.nn.MSELoss().cuda()
        self.l1_loss = torch.nn.L1Loss().cuda()

        # if (D_path):
        #     self.logger.info('load_D_path: ' + D_path)
        #     D_state_dict = torch.load(D_path)
        #     self.discriminator.load_state_dict(D_state_dict['D'])
        #     self.optimizer.load_state_dict(D_state_dict['D_optim'])

    def forward(self, fake, real):
        fake_detach = fake.detach()

        for _ in range(self.gan_k):
            self.optimizer.zero_grad()
            d_fake = self.discriminator(fake_detach)
            d_real = self.discriminator(real)
            if (self.gan_type.find('WGAN') >= 0):
                loss_d = (d_fake - d_real).mean()
                if self.gan_type.find('GP') >= 0:
                    epsilon = torch.rand(real.size(0), 1, 1, 1).cuda()
                    epsilon = epsilon.expand(real.size())
                    hat = fake_detach.mul(1 - epsilon) + real.mul(epsilon)
                    hat.requires_grad = True
                    d_hat = self.discriminator(hat)
                    gradients = torch.autograd.grad(
                        outputs=d_hat.sum(), inputs=hat,
                        retain_graph=True, create_graph=True, only_inputs=True
                    )[0]
                    gradients = gradients.view(gradients.size(0), -1)
                    gradient_norm = gradients.norm(2, dim=1)
                    gradient_penalty = 10 * gradient_norm.sub(1).pow(2).mean()
                    loss_d += gradient_penalty

            elif (self.gan_type == 'GAN'):
                valid_score = torch.ones(real.size(0), 1).cuda()
                fake_score = torch.zeros(real.size(0), 1).cuda()
                real_loss = self.bce_loss(torch.sigmoid(d_real), valid_score)
                fake_loss = self.bce_loss(torch.sigmoid(d_fake), fake_score)
                loss_d = (real_loss + fake_loss) / 2.
            elif (self.gan_type == 'LSGAN'):
                valid_score = torch.ones_like(d_real).cuda()
                fake_score = torch.zeros_like(d_fake).cuda()
                real_loss = self.mse_loss(d_real, valid_score)
                fake_loss = self.mse_loss(d_fake, fake_score)
                loss_d = (real_loss + fake_loss) / 2.
                # print("loss_d", loss_d)
            elif (self.gan_type == 'FM'):
                valid_score = torch.ones_like(d_real[-1]).cuda()
                fake_score = torch.zeros_like(d_fake[-1]).cuda()
                real_loss = self.mse_loss(d_real[-1], valid_score)
                fake_loss = self.mse_loss(d_fake[-1], fake_score)
                loss_d = (real_loss + fake_loss) / 2.
            # Discriminator update
            loss_d.backward()
            self.optimizer.step()
            self.lr_scheduler_f.step()

        d_fake_for_g = self.discriminator(fake)
        if (self.gan_type.find('WGAN') >= 0):
            loss_g = -d_fake_for_g.mean()
        elif (self.gan_type == 'GAN'):
            loss_g = self.bce_loss(torch.sigmoid(d_fake_for_g), valid_score)
        elif (self.gan_type == 'LSGAN'):
            loss_g = self.mse_loss(d_fake_for_g, valid_score)
        elif (self.gan_type == 'FM'):
            loss_g = 0.0
            for i in range(len(d_real)):
                if i == len(d_real) - 1:
                    break
                loss_g += self.l1_loss(d_fake_for_g[i], d_real[i].detach())
        # Generator loss
        return loss_g

    def state_dict(self):
        D_state_dict = self.discriminator.state_dict()
        D_optim_state_dict = self.optimizer.state_dict()
        lr_scheduler_f = self.lr_scheduler_f.state_dict()
        return D_state_dict, D_optim_state_dict, lr_scheduler_f

    def get_lr(self):
        return self.lr_scheduler_f.get_lr()

    def load_total(self, state_dict):
        self.discriminator.load_state_dict(state_dict[0])
        self.optimizer.load_state_dict(state_dict[1])
        self.lr_scheduler_f.load_state_dict(state_dict[2])

class CalculateLoss(nn.Module):
    def __init__(self, lr_config, train_patch, gan_type, use_perceptual_gan_loss):
        super().__init__()
        self.use_perceptual_gan_loss = use_perceptual_gan_loss
        if self.use_perceptual_gan_loss == True:
            self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            self.vgg = torchvision.models.vgg19(pretrained=True).features
            self.gan_loss = AdversarialLoss(lr_config=lr_config, train_crop_size=train_patch, gan_type=gan_type)
            total_params = sum(p.numel() for p in self.gan_loss.parameters())
            print(f'{total_params:,} gan total parameters.')
            total_trainable_params = sum(
                p.numel() for p in self.gan_loss.parameters() if p.requires_grad)
            print(f'{total_trainable_params:,} gan training parameters.')


    def forward(self, es, ta, isTrain, ref_align_mask = None):

        loss = [1 * torch.abs(es - ta).mean()]

        if self.use_perceptual_gan_loss == True:
            self.vgg = self.vgg.to(es.device)
            self.mean = self.mean.to(es.device)
            self.std = self.std.to(es.device)
            if isTrain == True:
                loss.append(5e-3 * self.gan_loss(es, ta))

                es = (es + 1) / 2
                ta = (ta + 1) / 2
                es = (es - self.mean) / self.std
                ta = (ta - self.mean) / self.std

                for midx, mod in enumerate(self.vgg):
                    es = mod(es)
                    with torch.no_grad():
                        ta = mod(ta)

                    if midx == 12:
                        lam = 1 * 0.01
                        loss.append(torch.abs(es - ta).mean() * lam)
                    elif midx == 21:
                        lam = 1 * 0.01
                        loss.append(torch.abs(es - ta).mean() * lam)
                    elif midx == 30:
                        lam = 1. * 0.01
                        loss.append(torch.abs(es - ta).mean() * lam)
                        break

        return loss

    def load_discriminator(self, state_path):
        state = torch.load(str(state_path))
        self.gan_loss.load_total(state["discriminator"])

    def get_discriminator(self):
        return self.gan_loss

class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

class RCAB(nn.Module):
    def __init__(
            self, conv, n_feat, kernel_size, reduction,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        res += x
        return res

class ResidualGroup(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, act, res_scale, n_resblocks):
        super(ResidualGroup, self).__init__()
        modules_body = [
            RCAB(
                conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True), res_scale=1) \
            for _ in range(n_resblocks)]
        modules_body.append(conv(n_feat, n_feat))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res

class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, bias=bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, bias=bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)

class MVSR_RCAN(nn.Module):
    def __init__(self, n_feats, res_scale, n_colors, conv=conv3x3):
        super(MVSR_RCAN, self).__init__()

        self.n_feats = n_feats
        act = nn.ReLU(True)
        self.act = act

        self.body_0_head = [nn.Conv2d(in_channels=n_colors, out_channels=n_feats, kernel_size=3, stride=1, padding=1)]
        self.body_0_head = nn.Sequential(*self.body_0_head)
        self.body_0 = [
            ResidualGroup(conv3x3, self.n_feats, 3, reduction=16, act=self.act, res_scale=res_scale, n_resblocks=10) for
            _ in range(1)]
        self.body_0 = nn.Sequential(*self.body_0)

        self.body_1_1 = [
            ResidualGroup(conv3x3, self.n_feats, 3, reduction=16, act=self.act, res_scale=res_scale, n_resblocks=8) for
            _ in range(1)]
        self.body_1_1 = nn.Sequential(*self.body_1_1)

        self.body_1_2 = [
            ResidualGroup(conv3x3, self.n_feats, 3, reduction=16, act=self.act, res_scale=res_scale, n_resblocks=8) for
            _ in range(1)]
        self.body_1_2 = nn.Sequential(*self.body_1_2)

        self.body_2_1 = [
            ResidualGroup(conv3x3, self.n_feats, 3, reduction=16, act=self.act, res_scale=res_scale, n_resblocks=6) for
            _ in range(1)]
        self.body_2_1 = nn.Sequential(*self.body_2_1)

        self.body_2_2 = [
            ResidualGroup(conv3x3, self.n_feats, 3, reduction=16, act=self.act, res_scale=res_scale, n_resblocks=6) for
            _ in range(1)]
        self.body_2_2 = nn.Sequential(*self.body_2_2)

        self.body_3_1 = [
            ResidualGroup(conv3x3, self.n_feats, 3, reduction=16, act=self.act, res_scale=res_scale, n_resblocks=4) for
            _ in range(1)]
        self.body_3_1 = nn.Sequential(*self.body_3_1)

        self.body_3_2 = [
            ResidualGroup(conv3x3, self.n_feats, 3, reduction=16, act=self.act, res_scale=res_scale, n_resblocks=4) for
            _ in range(1)]
        self.body_3_2 = nn.Sequential(*self.body_3_2)

        ###########################################
        #################### RSM ##################
        ###########################################

        self.body_1_head = nn.Conv2d(in_channels=n_feats * 5, out_channels=n_feats, kernel_size=3, stride=1,
                                     padding=1)
        self.body_2_head = nn.Conv2d(in_channels=n_feats * 3, out_channels=n_feats, kernel_size=3, stride=1,
                                     padding=1)
        self.body_3_head = nn.Conv2d(in_channels=n_feats * 2, out_channels=n_feats, kernel_size=3, stride=1,
                                     padding=1)
        self.body_1_select = [
            nn.Conv2d(in_channels=n_feats * 3, out_channels=n_feats, kernel_size= 3, stride=1, padding=1),
            nn.LeakyReLU(True),
            nn.Conv2d(in_channels=n_feats, out_channels= 2, kernel_size=3, stride=1, padding=1),
        ]
        self.body_1_select = nn.Sequential(*self.body_1_select)

        self.body_2_select = [
            nn.Conv2d(in_channels=n_feats * 3, out_channels=n_feats, kernel_size= 3, stride=1, padding=1),
            nn.LeakyReLU(True),
            nn.Conv2d(in_channels=n_feats, out_channels= 2, kernel_size=3, stride=1, padding=1),
        ]
        self.body_2_select = nn.Sequential(*self.body_2_select)

        self.body_3_select = [
            nn.Conv2d(in_channels=n_feats * 3, out_channels=n_feats, kernel_size= 3, stride=1, padding=1),
            nn.LeakyReLU(True),
            nn.Conv2d(in_channels=n_feats, out_channels= 2, kernel_size=3, stride=1, padding=1),
        ]
        self.body_3_select = nn.Sequential(*self.body_3_select)

        ###########################################
        #################### ASM ##################
        ###########################################

        self.body_1_head_2_1 = nn.Conv2d(in_channels=n_feats * 5, out_channels=n_feats * 4, kernel_size=3, stride=1,
                                     padding=1)
        self.body_1_head_2_2 = nn.Conv2d(in_channels=n_feats * 4, out_channels=n_feats, kernel_size=3, stride=1,
                                     padding=1)
        self.body_2_head_2_1 = nn.Conv2d(in_channels=n_feats * 3, out_channels=n_feats * 2, kernel_size=3, stride=1,
                                     padding=1)
        self.body_2_head_2_2 = nn.Conv2d(in_channels=n_feats * 2, out_channels=n_feats, kernel_size=3, stride=1,
                                     padding=1)
        self.body_3_head_2_1 = nn.Conv2d(in_channels=n_feats * 2, out_channels=n_feats, kernel_size=3, stride=1,
                                     padding=1)
        self.body_3_head_2_2 = nn.Conv2d(in_channels=n_feats, out_channels=n_feats, kernel_size=3, stride=1,
                                     padding=1)

        self.body_1_3 = [
            ResidualGroup(conv3x3, self.n_feats, 3, reduction=16, act=self.act, res_scale=res_scale, n_resblocks=8)
            for
            _ in range(1)]
        self.body_1_3 = nn.Sequential(*self.body_1_3)
        self.body_2_3 = [
            ResidualGroup(conv3x3, self.n_feats, 3, reduction=16, act=self.act, res_scale=res_scale, n_resblocks=6)
            for
            _ in range(1)]
        self.body_2_3 = nn.Sequential(*self.body_2_3)
        self.body_3_3 = [
            ResidualGroup(conv3x3, self.n_feats, 3, reduction=16, act=self.act, res_scale=res_scale, n_resblocks=4)
            for
            _ in range(1)]
        self.body_3_3 = nn.Sequential(*self.body_3_3)

        self.body_1_head_3 = nn.Conv2d(in_channels=n_feats * 2, out_channels=n_feats, kernel_size=3, stride=1,
                                     padding=1)
        self.body_2_head_3 = nn.Conv2d(in_channels=n_feats * 2, out_channels=n_feats, kernel_size=3, stride=1,
                                     padding=1)
        self.body_3_head_3 = nn.Conv2d(in_channels=n_feats * 2, out_channels=n_feats, kernel_size=3, stride=1,
                                     padding=1)
        self.body_1_select_3 = [
            nn.Conv2d(in_channels=n_feats * 3, out_channels=n_feats, kernel_size= 3, stride=1, padding=1),
            nn.LeakyReLU(True),
            nn.Conv2d(in_channels=n_feats, out_channels= 2, kernel_size=3, stride=1, padding=1),
        ]
        self.body_1_select_3 = nn.Sequential(*self.body_1_select_3)

        self.body_2_select_3 = [
            nn.Conv2d(in_channels=n_feats * 3, out_channels=n_feats, kernel_size= 3, stride=1, padding=1),
            nn.LeakyReLU(True),
            nn.Conv2d(in_channels=n_feats, out_channels= 2, kernel_size=3, stride=1, padding=1),
        ]
        self.body_2_select_3 = nn.Sequential(*self.body_2_select_3)

        self.body_3_select_3 = [
            nn.Conv2d(in_channels=n_feats * 3, out_channels=n_feats, kernel_size= 3, stride=1, padding=1),
            nn.LeakyReLU(True),
            nn.Conv2d(in_channels=n_feats, out_channels= 2, kernel_size=3, stride=1, padding=1),
        ]
        self.body_3_select_3 = nn.Sequential(*self.body_3_select_3)


        self.body_1_last = [
            Upsampler(conv, 2, n_feats, act=False),
            conv(n_feats, n_feats)
        ]
        self.body_1_last = nn.Sequential(*self.body_1_last)
        self.body_2_last = [
            Upsampler(conv, 2, n_feats, act=False),
            conv(n_feats, n_feats)
        ]
        self.body_2_last = nn.Sequential(*self.body_2_last)
        self.last = nn.Conv2d(in_channels=n_feats, out_channels=n_colors, kernel_size=3, stride=1, padding=1)


    def forward(self, img_lr, img_ref_multiview, img_ref_near):
        img_lr = self.body_0(self.body_0_head(img_lr))

        # x1
        # rsm
        img_ref_multiview_tmp = img_lr.clone() + self.body_1_head(torch.cat([img_lr, img_ref_multiview[0]], dim=1))
        img_ref_near_tmp = img_lr.clone() + self.body_1_head(torch.cat([img_lr, img_ref_near[0]], dim=1))
        img_ref_alpha_1 = self.body_1_select(torch.cat([img_lr, img_ref_multiview_tmp, img_ref_near_tmp], dim=1))
        img_ref_alpha_1 = torch.softmax(img_ref_alpha_1, dim=1)
        img_lr = img_ref_multiview_tmp * img_ref_alpha_1[:, 0:1, ...] + img_ref_near_tmp * img_ref_alpha_1[:, 1:2, ...]
        img_lr = self.body_1_1(img_lr)

        # asm
        img_ref_multiview_tmp = img_ref_multiview[0].clone() + self.body_1_head_2_1(torch.cat([img_ref_multiview[0], img_lr], dim=1))
        img_ref_multiview_tmp = self.body_1_head_2_2(img_ref_multiview_tmp)
        img_ref_multiview_tmp = self.body_1_3(img_ref_multiview_tmp)
        img_ref_near_tmp = img_ref_near[0].clone() + self.body_1_head_2_1(torch.cat([img_ref_near[0], img_lr], dim=1))
        img_ref_near_tmp = self.body_1_head_2_2(img_ref_near_tmp)
        img_ref_near_tmp = self.body_1_3(img_ref_near_tmp)

        img_ref_multiview_tmp = img_lr.clone() + self.body_1_head_3(torch.cat([img_lr, img_ref_multiview_tmp], dim=1))
        img_ref_near_tmp = img_lr.clone() + self.body_1_head_3(torch.cat([img_lr, img_ref_near_tmp], dim=1))
        img_ref_alpha_1_3 = self.body_1_select_3(torch.cat([img_lr, img_ref_multiview_tmp, img_ref_near_tmp], dim=1))
        img_ref_alpha_1_3 = torch.softmax(img_ref_alpha_1_3, dim=1)
        img_lr = img_ref_multiview_tmp * img_ref_alpha_1_3[:, 0:1, ...] + img_ref_near_tmp * img_ref_alpha_1_3[:, 1:2, ...]
        img_lr = self.body_1_last(self.body_1_2(img_lr))

        # x2
        # rsm
        img_ref_multiview_tmp = img_lr.clone() + self.body_2_head(torch.cat([img_lr, img_ref_multiview[1]], dim=1))
        img_ref_near_tmp = img_lr.clone() + self.body_2_head(torch.cat([img_lr, img_ref_near[1]], dim=1))
        img_ref_alpha_2 = self.body_2_select(torch.cat([img_lr, img_ref_multiview_tmp, img_ref_near_tmp], dim=1))
        img_ref_alpha_2 = torch.softmax(img_ref_alpha_2, dim=1)
        img_lr = img_ref_multiview_tmp * img_ref_alpha_2[:, 0:1, ...] + img_ref_near_tmp * img_ref_alpha_2[:, 1:2, ...]
        img_lr = self.body_2_1(img_lr)

        # asm
        img_ref_multiview_tmp = img_ref_multiview[1].clone() + self.body_2_head_2_1(torch.cat([img_ref_multiview[1], img_lr], dim=1))
        img_ref_multiview_tmp = self.body_2_head_2_2(img_ref_multiview_tmp)
        img_ref_multiview_tmp = self.body_2_3(img_ref_multiview_tmp)
        img_ref_near_tmp = img_ref_near[1].clone() + self.body_2_head_2_1(torch.cat([img_ref_near[1], img_lr], dim=1))
        img_ref_near_tmp = self.body_2_head_2_2(img_ref_near_tmp)
        img_ref_near_tmp = self.body_2_3(img_ref_near_tmp)

        img_ref_multiview_tmp = img_lr.clone() + self.body_2_head_3(torch.cat([img_lr, img_ref_multiview_tmp], dim=1))
        img_ref_near_tmp = img_lr.clone() + self.body_2_head_3(torch.cat([img_lr, img_ref_near_tmp], dim=1))
        img_ref_alpha_2_3 = self.body_2_select_3(torch.cat([img_lr, img_ref_multiview_tmp, img_ref_near_tmp], dim=1))
        img_ref_alpha_2_3 = torch.softmax(img_ref_alpha_2_3, dim=1)
        img_lr = img_ref_multiview_tmp * img_ref_alpha_2_3[:, 0:1, ...] + img_ref_near_tmp * img_ref_alpha_2_3[:, 1:2, ...]
        img_lr = self.body_2_last(self.body_2_2(img_lr))


        # x4
        # rsm
        img_ref_multiview_tmp = img_lr.clone() + self.body_3_head(torch.cat([img_lr, img_ref_multiview[2]], dim=1))
        img_ref_near_tmp = img_lr.clone() + self.body_3_head(torch.cat([img_lr, img_ref_near[2]], dim=1))
        img_ref_alpha_3 = self.body_3_select(torch.cat([img_lr, img_ref_multiview_tmp, img_ref_near_tmp], dim=1))
        img_ref_alpha_3 = torch.softmax(img_ref_alpha_3, dim=1)
        img_lr = img_ref_multiview_tmp * img_ref_alpha_3[:, 0:1, ...] + img_ref_near_tmp * img_ref_alpha_3[:, 1:2, ...]
        img_lr = self.body_3_1(img_lr)

        # asm
        img_ref_multiview_tmp = img_ref_multiview[2].clone() + self.body_3_head_2_1(torch.cat([img_ref_multiview[2], img_lr], dim=1))
        img_ref_multiview_tmp = self.body_3_head_2_2(img_ref_multiview_tmp)
        img_ref_multiview_tmp = self.body_3_3(img_ref_multiview_tmp)
        img_ref_near_tmp = img_ref_near[2].clone() + self.body_3_head_2_1(torch.cat([img_ref_near[2], img_lr], dim=1))
        img_ref_near_tmp = self.body_3_head_2_2(img_ref_near_tmp)
        img_ref_near_tmp = self.body_3_3(img_ref_near_tmp)

        img_ref_multiview_tmp = img_lr.clone() + self.body_3_head_3(torch.cat([img_lr, img_ref_multiview_tmp], dim=1))
        img_ref_near_tmp = img_lr.clone() + self.body_3_head_3(torch.cat([img_lr, img_ref_near_tmp], dim=1))
        img_ref_alpha_3_3 = self.body_3_select_3(torch.cat([img_lr, img_ref_multiview_tmp, img_ref_near_tmp], dim=1))
        img_ref_alpha_3_3 = torch.softmax(img_ref_alpha_3_3, dim=1)
        img_lr = img_ref_multiview_tmp * img_ref_alpha_3_3[:, 0:1, ...] + img_ref_near_tmp * img_ref_alpha_3_3[:, 1:2, ...]
        img_lr = self.last(self.body_3_2(img_lr))

        return img_lr, [img_ref_alpha_1, img_ref_alpha_2, img_ref_alpha_3, img_ref_alpha_1_3, img_ref_alpha_2_3, img_ref_alpha_3_3]

class Feature_Extractor(nn.Module):

    def __init__(self, nf=64, ref_num=6):
        super(Feature_Extractor, self).__init__()
        self.nf = nf
        act = nn.ReLU(True)
        self.act = act
        self.ref_num = ref_num
        self.body_1_head = nn.Conv2d(in_channels= 3 * (self.ref_num + 1), out_channels=self.nf, kernel_size=3, stride=1, padding=1)
        self.body_1 = [ResidualGroup(conv3x3, self.nf, 3, reduction=16, act = self.act, res_scale=1, n_resblocks=3) for
                     _ in range(1)]
        self.body_1 = nn.Sequential(*self.body_1)

        self.body_2_head = nn.Conv2d(in_channels=self.nf, out_channels=self.nf*2, kernel_size=4, stride=2, padding=1)
        self.body_2 = [ResidualGroup(conv3x3, self.nf*2, 3, reduction=16, act = self.act, res_scale=1, n_resblocks=3) for
                     _ in range(1)]
        self.body_2 = nn.Sequential(*self.body_2)

        self.body_3_head = nn.Conv2d(in_channels=self.nf*2, out_channels=self.nf*4, kernel_size=4, stride=2, padding=1)
        self.body_3 = [ResidualGroup(conv3x3, self.nf*4, 3, reduction=16, act = self.act, res_scale=1, n_resblocks=3) for
                     _ in range(1)]
        self.body_3 = nn.Sequential(*self.body_3)

    def forward(self, ref):
        ref_1 = self.body_1(self.body_1_head(ref))
        ref_2 = self.body_2(self.body_2_head(ref_1))
        ref_3 = self.body_3(self.body_3_head(ref_2))
        return [ref_3, ref_2, ref_1]

class MVSRnet(nn.Module):
    def __init__(
            self,
            nf=64,
            ref_num=6,
    ):
        super().__init__()
        self.nf = nf
        self.ref_num = ref_num

        self.feature_extract = Feature_Extractor(self.nf, self.ref_num)

        self.mvsr = MVSR_RCAN(n_feats=64, res_scale=1, n_colors=3, conv=conv3x3)

    def forward_train(self, **kwargs):
        img_lr = kwargs["img_lr"].float()
        img_bic = kwargs["img_bic"].float()
        img_ref_multiview = kwargs["img_ref_multiview"].float()
        img_ref_near = kwargs["img_ref_near"].float()
        bs, n, c_rgb, h, w = img_ref_multiview.shape

        img_ref_multiview = img_ref_multiview.reshape(bs, n * c_rgb, h, w)
        img_ref_multiview = torch.cat([img_ref_multiview, img_bic], dim=1)
        img_ref_multiview = self.feature_extract(img_ref_multiview)

        img_ref_near = img_ref_near.reshape(bs, n * c_rgb, h, w)
        img_ref_near = torch.cat([img_ref_near, img_bic], dim=1)
        img_ref_near = self.feature_extract(img_ref_near)

        hr, ref_align_mask = self.mvsr(img_lr, img_ref_multiview, img_ref_near)

        hr = torch.clamp(hr, -1, 1)
        return {"out": hr, 'ref_align_mask': ref_align_mask}

    def forward(self, **kwargs):
        return self.forward_train(**kwargs)

def get_mvsr_net(nf, ref_num):
    model = MVSRnet(nf=nf, ref_num=ref_num)
    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')
    # exit(0)
    return model

