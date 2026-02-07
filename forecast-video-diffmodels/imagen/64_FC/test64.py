import glob
import copy
import argparse
from functools import partial, partialmethod
from einops import rearrange
import os

import sys
sys.path.append("../")
sys.path.append("../imagen/")
sys.path.append("../../dataproc/")
from utils import sample
from helpers import *
try:
    from imagen_api_compat import Unet3D, Imagen, ImagenTrainer, NullUnet
except ImportError:
    from imagen_pytorch import Unet3D, Imagen, ImagenTrainer, NullUnet
from send_emails import *

try:
    import wandb
except Exception:
    wandb = None

seed_value = 42
torch.manual_seed(seed_value)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed_value)

parser = argparse.ArgumentParser()
parser.add_argument('-run_name', help='Specify the run name (for eg. 64_FC_3e-4)')
parser.add_argument('-best_epoch', help='best epoch')
parser.add_argument('--log_to_file', action='store_true', help='redirect stdout to metrics log file')
parser.add_argument('--disable_tqdm', action='store_true', help='disable tqdm progress bars')
parser.add_argument('--enable_wandb', action='store_true', help='enable Weights & Biases logging')
parser.add_argument('--wandb_project', default='cyclone-forecasting', help='wandb project')
parser.add_argument('--wandb_entity', default=None, help='wandb entity/team')
parser.add_argument('--wandb_run_name', default=None, help='wandb run name')
parser.add_argument('--wandb_mode', default='online', choices=['online', 'offline', 'disabled'], help='wandb mode')
args = parser.parse_args()

RUN_NAME = args.run_name
BEST_EPOCH = args.best_epoch
if args.log_to_file:
    sys.stdout = open(f'{RUN_NAME}_TEST_METRICS_LOG_{datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")}.log','wt')
print = partial(print, flush=True)
tqdm_disabled = args.disable_tqdm or os.environ.get("FDM_DISABLE_TQDM", "0") == "1"
tqdm.__init__ = partialmethod(tqdm.__init__, disable=tqdm_disabled)


BASE_DIR = f"{BASE_HOME}/models/{RUN_NAME}/models/{RUN_NAME}/"

print(f"Run name: {RUN_NAME}")

best_epoch_dict = {
    "64_FC_rot904_sep_3e-4": 180,
    "64_FC_rot904_3e-4": 240,
    "64_FC_3e-4": 235,
    "v_64_FC_3e-4": 390,
    "v_64_FC_3e-4_dim64": 390,
    "v_64_FC_3e-4_dim64_img": 390,
    "v_64_FC_3e-4_dim128": 340,
    "v_64_FC_3e-4_dim256": 99,
    "v_64_FC_3e-4_dim256_2048": 99,
    "v_64_FC_3e-4_dim_2048": 99
}

unets, O_SIZE = run_name_info(RUN_NAME)

class DDPMArgs:
    def __init__(self):
        pass
    
args = DDPMArgs()
args.batch_size = 1
args.image_size = O_SIZE ; args.o_size = O_SIZE ; args.n_size = 128 ;
args.continuous_embed_dim = args.o_size*args.o_size*3*10
args.dataset_path = f"{BASE_DATA}/satellite/dataloader/{args.o_size}_FC"
args.datalimit = False
args.lr = 3e-4
args.mode = "fc"

train_dataloader, test_dataloader = get_satellite_data(args, "vid")
test_dataloader.switch_to_vid()
_ = len(train_dataloader) ; _ = len(test_dataloader)

print("Dataloaders loaded.")

if '1k' in RUN_NAME:
    timesteps = 1000
else:
    timesteps = 250

# imagen = Imagen(
#     unets = unets,
#     image_sizes = (args.image_size),
#     timesteps = 250,
#     cond_drop_prob = 0.1,
#     condition_on_continuous = True,
#     continuous_embed_dim = args.continuous_embed_dim,
# )

metric_dict = {
    "kl_div": [],
    "rmse": [],
    "mae":  [],
    "psnr": [],
    "ssim": [],
    "fid": [],
    "fvd": [],
    "ssim2": [],
    "psnr2": [],
    "lpips": []
}

test_metric_dict = copy.deepcopy(metric_dict)
best_epoch = BEST_EPOCH#best_epoch_dict[RUN_NAME]
ckpt_trainer_path = f"{BASE_DIR}/ckpt_trainer_1_{best_epoch:03}.pt"
# trainer = ImagenTrainer(imagen, lr=args.lr, verbose=False).cuda()
# trainer.load(ckpt_trainer_path)

wandb_run = None
if args.enable_wandb:
    if wandb is None:
        print("[wandb] not installed, continuing without wandb")
    else:
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name or f"{RUN_NAME}-test",
            mode=args.wandb_mode,
            config={"run_name": RUN_NAME, "best_epoch": int(BEST_EPOCH)}
        )
        print(f"[wandb] initialized run: {wandb_run.name}")

for idx in range(len(test_dataloader)):
    print(f"Evaluating batch idx {idx} ...")

    y_true, y_pred = sample(test_dataloader, RUN_NAME, idx, args, ckpt_trainer_path)
    metric_dict = calculate_metrics(y_pred, y_true)
    for key in metric_dict.keys():
        test_metric_dict[key].append(metric_dict[key])
    if wandb_run is not None:
        for key, value in metric_dict.items():
            val = float(value.detach().item() if torch.is_tensor(value) else value)
            wandb.log({f"test/{key}": val}, step=idx)

print("individual")
print(test_metric_dict)
# Initialize a dictionary to store the averages
average_metric_dict = {}

# Iterate over each key in the dictionary
for key, values in test_metric_dict.items():
    # Calculate the average for the current key
    average_metric_dict[key] = sum(values) / len(values)

# Now average_metric_dict will contain the average values for each key
print("average")
print(average_metric_dict)
if wandb_run is not None:
    wandb.log({f"test/avg_{k}": float(v.detach().item() if torch.is_tensor(v) else v) for k, v in average_metric_dict.items()})
    wandb.finish()

with open(f"{BASE_HOME}/models/{RUN_NAME}/metrics_test.pkl", "wb") as file:
    pickle.dump(test_metric_dict, file)

print(f'Evaluation completed.')

subject = f"[COMPLETED] Test Evaluation Metrics"
message_txt = f"""Metrics Evaluation Completed for {RUN_NAME}"""
send_txt_email(message_txt, subject)
