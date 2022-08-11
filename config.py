from pathlib import Path
import socket
import platform
import getpass
import os

HOSTNAME = socket.gethostname()
PLATFORM = platform.system()
USER = getpass.getuser()

use_cuda = True

gtav = False
blendedmvs = False
tanks_temples_root = "../GARS/Tanks_and_Temples_HLBIC_FR_MVRs_NVRs/LR/x4/"
gtav_root = "../GARS/GTAV_720_HLBIC_FR_MVRs_NVRs/LR/x4/"
blendedmvs_root = "../GARS/BlendedMVS_HLBIC_FR_MVRs_NVRs/LR/x4/"

tat_root = Path(tanks_temples_root)
if gtav == True:
    tat_root = Path(gtav_root)
if blendedmvs == True:
    tat_root = Path(blendedmvs_root)


train_iters = 200000
lr = 1e-4
eta_min = 1e-7

# loss
pad_width = 4
train_batch_size = 2
eval_batch_size = 1
train_patch = [80, 160]
use_perceptual_gan_loss = True

# datasets
use_complement = True
ref_num = 6

experiment_name = f"backbone"

if use_perceptual_gan_loss == True:
    experiment_name = experiment_name + '_perceptual_gan'

# datasets
experiment_name = experiment_name + '_ref_' + str(ref_num)
experiment_name = experiment_name + '_patch_' + str(train_patch)
if use_complement == True:
    experiment_name = experiment_name + '_complement'



scale = 4
pin_memory = True
manul_save = 5000
manul_save_model = 10000

nf = 64
beta1 = 0.9
beta2 = 0.999
T_period = [train_iters, train_iters]
restarts = [train_iters]
restart_weights = [1]

gan_type = "WGAN_GP"
loss_begin = 0
T_period_gan = [train_iters-loss_begin, train_iters-loss_begin]
restarts_gan = [train_iters-loss_begin]
restart_weights_gan = [1]


tat_train_sets = [
    "training/Barn",
    "training/Caterpillar",
    "training/Church",
    "training/Courthouse",
    "training/Ignatius",
    "training/Meetingroom",
    "intermediate/Family",
    "intermediate/Francis",
    "intermediate/Horse",
    "intermediate/Lighthouse",
    "intermediate/Panther",
    "advanced/Auditorium",
    "advanced/Ballroom",
    "advanced/Museum",
    "advanced/Temple",
    "advanced/Courtroom",
    "advanced/Palace",
]
tat_eval_sets = [
    "intermediate/Train",
    "intermediate/M60",
    "training/Truck",
    "intermediate/Playground",
]


if gtav ==True:
    tat_train_sets = []
    tat_eval_sets = []
    for i in range(120):
        tat_train_sets.append(str(i).zfill(4))
        tat_eval_sets.append(str(i).zfill(4))

if blendedmvs == True:
    tat_train_sets = os.listdir(blendedmvs_root)
    tat_train_sets.sort()
    tat_eval_sets = os.listdir(blendedmvs_root)
    tat_eval_sets.sort()


