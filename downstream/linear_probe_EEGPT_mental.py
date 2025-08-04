import random 
import os
from typing import Any, Optional
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
from torch import nn
import pytorch_lightning as pl

from functools import partial
import numpy as np
import tqdm
from pytorch_lightning import loggers as pl_loggers
import torch.nn.functional as F

# import wandb
import statistics
from glob import glob

def seed_torch(seed=1029):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True
seed_torch(7)

from Modules.models.EEGPT_mcae import EEGTransformer

from Modules.Network.utils import Conv1dWithConstraint, LinearWithConstraint
from sklearn import metrics
from utils_eval import get_metrics

use_channels_names = ['FP1', 'FP2', 'F3', 'FZ', 'F4', 'F7', 'F8', 'T7', 'T8', 'C3', 'Cz', 'C4', 'P7', 'P8', 'P3', 'PZ', 'P4', 'O1', 'O2']
# CHANNEL_DICT = {k.upper():v for v,k in enumerate(
#                      [      'FP1', 'FPZ', 'FP2', 
#                         "AF7", 'AF3', 'AF4', "AF8", 
#             'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 
#         'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 
#             'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 
#         'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8',
#              'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 
#                       'PO7', "PO5", 'PO3', 'POZ', 'PO4', "PO6", 'PO8', 
#                                'O1', 'OZ', 'O2', ])}

class LitEEGPTCausal(pl.LightningModule):

    def __init__(self, load_path="../checkpoint/eegpt_mcae_58chs_4s_large4E.ckpt", num_classes=6):
        super().__init__()    
        self.chans_num = len(use_channels_names)
        # init model
        target_encoder = EEGTransformer(
            img_size=[self.chans_num, int(20 * 256)],
            patch_size=32*2,
            patch_stride = 32,
            embed_num=4,
            embed_dim=512,
            depth=8,
            num_heads=8,
            mlp_ratio=4.0,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            init_std=0.02,
            qkv_bias=True, 
            norm_layer=partial(nn.LayerNorm, eps=1e-6))
            
        self.target_encoder = target_encoder
        for param in self.target_encoder.parameters():
            param.requires_grad = False
        # self.predictor      = predictor
        self.chans_id       = target_encoder.prepare_chan_ids(use_channels_names)
        
        # -- load checkpoint
        pretrain_ckpt = torch.load(load_path)
        
        target_encoder_stat = {}
        for k,v in pretrain_ckpt['state_dict'].items():
            if k.startswith("target_encoder."):
                target_encoder_stat[k[15:]]=v
        
        self.chan_scale       = torch.nn.Parameter(torch.ones(1, self.chans_num, 1)+ 0.001 * torch.rand((1, self.chans_num, 1)), requires_grad=True)
        
                
        self.target_encoder.load_state_dict(target_encoder_stat)
        
        self.linear_probe1   =   LinearWithConstraint(2048, 16, max_norm=1)
        self.linear_probe2   =   LinearWithConstraint(2544, num_classes, max_norm=0.25)
       
        self.drop           = torch.nn.Dropout(p=0.50)
        
        self.loss_fn        = torch.nn.CrossEntropyLoss()
        
        self.running_scores = {"train":[], "valid":[], "test":[]}
        self.is_sanity = True
        
    
    def forward(self, x):
        # print(x.shape) # B, C, T
        B, C, T = x.shape
        
        x = x.to(torch.float)
        
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True) + 1e-8
        x = (x - mean) / std
        
        # x = x[:,channels_index,:]
        x = x * self.chan_scale
        
        self.target_encoder.eval()
        z = self.target_encoder(x, self.chans_id.to(x))
        
        h = z.flatten(2)
        h = self.linear_probe1(self.drop(h))
        
        h = h.flatten(1)
        h = self.linear_probe2(h)
        
        return x, h
    def on_train_epoch_start(self) -> None:
        torch.cuda.empty_cache()
        self.running_scores["train"]=[]
        return super().on_train_epoch_start()
    
    def on_train_epoch_end(self) -> None:
        label, y_score, pred = [], [], []
        for x, y, z in self.running_scores["train"]:
            label.append(x)
            y_score.append(y)
            pred.append(z)
        label = torch.cat(label, dim=0)
        y_score = torch.cat(y_score, dim=0)
        pred = torch.cat(pred, dim=0)
        f1 = metrics.f1_score(label, pred, average='macro')
        self.log('train_f1', f1, on_epoch=True, on_step=False, sync_dist=True)
        return super().on_train_epoch_end()
    
    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x, y = batch
        label = y.long()
        
        x, logit = self.forward(x)
        loss = self.loss_fn(logit, label)
        preds = torch.argmax(logit, dim=-1)
        accuracy = ((preds==label)*1.0).mean()
        y_score =  logit
        y_score =  torch.softmax(y_score, dim=-1)[:,1]
        self.running_scores["train"].append((label.clone().detach().cpu(), y_score.clone().detach().cpu(), preds.clone().detach().cpu()))
        
        # Logging to TensorBoard by default
        self.log('train_loss', loss, on_epoch=True, on_step=False, sync_dist=True)
        self.log('train_acc', accuracy, on_epoch=True, on_step=False, sync_dist=True)
        self.log('data_avg', x.mean(), on_epoch=True, on_step=False, sync_dist=True)
        self.log('data_max', x.max(), on_epoch=True, on_step=False, sync_dist=True)
        self.log('data_min', x.min(), on_epoch=True, on_step=False, sync_dist=True)
        self.log('data_std', x.std(), on_epoch=True, on_step=False, sync_dist=True)
        
        return loss
    def on_validation_epoch_start(self) -> None:
        self.running_scores["valid"]=[]
        return super().on_validation_epoch_start()
    
    def on_validation_epoch_end(self) -> None:
        if self.is_sanity:
            self.is_sanity=False
            return super().on_validation_epoch_end()
            
        label, y_score = [], []
        for x,y in self.running_scores["valid"]:
            label.append(x)
            y_score.append(y)
        label = torch.cat(label, dim=0)
        y_score = torch.cat(y_score, dim=0)
        print(label.shape, y_score.shape)
        
        metrics = ["accuracy", "balanced_accuracy", "cohen_kappa", "roc_auc_macro_ovr", "f1_macro", "roc_auc_macro_ovo"]
        results = get_metrics(y_score.cpu().numpy(), label.cpu().numpy(), metrics, False)
        
        for key, value in results.items():
            self.log('valid_'+key, value, on_epoch=True, on_step=False, sync_dist=True)
        return super().on_validation_epoch_end()
    
    def validation_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x, y = batch
        label = y.long()
        
        x, logit = self.forward(x)
        
        preds = torch.argmax(logit, dim=-1)
        accuracy = ((preds==label)*1.0).mean()
        
        loss = self.loss_fn(logit, label)
        y_score =  logit
        y_score =  torch.softmax(y_score, dim=-1)
        self.running_scores["valid"].append((label.clone().detach().cpu(), y_score.clone().detach().cpu()))
        # Logging to TensorBoard by default
        self.log('valid_loss', loss, on_epoch=True, on_step=False, sync_dist=True)
        self.log('valid_acc', accuracy, on_epoch=True, on_step=False, sync_dist=True)
        return loss
    
    def configure_optimizers(self):
        
        optimizer = torch.optim.AdamW(
            [self.chan_scale]+
            list(self.linear_probe1.parameters())+
            list(self.linear_probe2.parameters()),
            weight_decay=0.01)
        
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr, steps_per_epoch=steps_per_epoch, epochs=max_epochs, pct_start=0.2)
        lr_dict = {
            'scheduler': lr_scheduler, # The LR scheduler instance (required)
            # The unit of the scheduler's step size, could also be 'step'
            'interval': 'step',
            'frequency': 1, # The frequency of the scheduler
            'monitor': 'val_loss', # Metric for `ReduceLROnPlateau` to monitor
            'strict': True, # Whether to crash the training if `monitor` is not found
            'name': None, # Custom name for `LearningRateMonitor` to use
        }
      
        return (
            {'optimizer': optimizer, 'lr_scheduler': lr_dict},
        )
        # return {'optimizer': optimizer}
        
# load configs
# -- LOSO 

# load configs
import torchvision
import math


global max_epochs
global steps_per_epoch
global max_lr

batch_size = 64
max_epochs = 50



all_subjects = range(510)
train_ids = [0, 1, 2, 3, 4, 7, 8, 9, 11, 12, 13, 16, 17, 18, 19, 20, 21, 22, 24, 25, 27, 28, 29, 32, 33, 34, 35, 36, 37, 39, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 56, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 69, 70, 71, 73, 74, 76, 77, 78, 79, 80, 81, 82, 83, 87, 88, 89, 90, 92, 93, 94, 95, 96, 97, 98, 100, 101, 102, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 116, 118, 119, 120, 121, 122, 123, 125, 126, 128, 129, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 146, 147, 148, 149, 152, 153, 155, 156, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 170, 173, 174, 176, 177, 178, 179, 181, 182, 183, 184, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 199, 200, 202, 203, 204, 205, 206, 207, 208, 209, 211, 214, 215, 216, 217, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 235, 236, 237, 239, 240, 241, 242, 243, 244, 245, 247, 248, 251, 252, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 269, 270, 272, 273, 274, 276, 277, 278, 281, 282, 283, 284, 285, 286, 287, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 305, 307, 308, 309, 310, 312, 313, 314, 315, 316, 317, 319, 320, 321, 322, 323, 324, 325, 326, 327, 329, 330, 332, 333, 335, 336, 338, 339, 340, 342, 343, 345, 350, 351, 353, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 366, 367, 370, 372, 373, 374, 375, 376, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 391, 392, 393, 395, 396, 397, 398, 401, 402, 404, 406, 408, 409, 410, 412, 413, 414, 415, 416, 417, 418, 419, 421, 422, 423, 424, 425, 426, 427, 428, 429, 431, 432, 433, 434, 435, 436, 437, 438, 440, 441, 442, 443, 444, 445, 447, 448, 449, 450, 451, 454, 455, 456, 457, 458, 459, 460, 461, 462, 463, 464, 465, 466, 467, 469, 470, 471, 472, 473, 474, 475, 477, 478, 480, 481, 482, 483, 484, 485, 487, 488, 489, 491, 492, 493, 494, 497, 500, 501, 502, 503, 506, 507, 508, 509, 510]
test_ids = [k for k in all_subjects if k not in train_ids]

sub_train = [f".sub{x}" for x in train_ids]
sub_valid = [f".sub{x}" for x in test_ids]

train_dataset = torchvision.datasets.DatasetFolder(root="/mnt/s3-data2/amiftakhova/ocd_eegpt", loader=torch.load, extensions=sub_train)
valid_dataset = torchvision.datasets.DatasetFolder(root="/mnt/s3-data2/amiftakhova/ocd_eegpt", loader=torch.load, extensions=sub_valid)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=8, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, num_workers=8, shuffle=False)

steps_per_epoch = math.ceil(len(train_loader))

# init model
model = LitEEGPTCausal()

# most basic trainer, uses good defaults (auto-tensorboard, checkpoints, logs, and more)
lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')
callbacks = [lr_monitor]
max_lr = 8e-4

# wb_logger = pl_loggers.WandbLogger(
#     project="EEGPT_mental",  # Your W&B project name
#     name="EEGPT_mental_v4",  # Experiment name
#     log_model=False,        # Set to True if you want to log model checkpoints
#     save_dir="./logs/"
# )

trainer = pl.Trainer(accelerator='cuda',
            max_epochs=max_epochs, 
            callbacks=callbacks,
            enable_checkpointing=False,
            logger=[pl_loggers.TensorBoardLogger('./logs/', name="EEGPT_mental", version=f"v1"), 
                    pl_loggers.CSVLogger('./logs/', name="EEGPT_mental_csv"),
                    # wb_logger
                    ])

trainer.fit(model, train_loader, valid_loader)
trainer.save_checkpoint('/mnt/s3-data2/amiftakhova/eegpt_ckpts/EEGPT_mental_v2.ckpt')
# wandb.finish()

# CALCULATE METRICS ON TEST SET
model = model.eval()
files = [x for x in glob("/mnt/s3-data2/amiftakhova/ocd_eegpt/*/*.sub*") if any([x.endswith(f".sub{y}") for y in test_ids])]
by_subj = {}
for f in files:
    subj = f.split('.sub')[-1]
    if subj not in by_subj:
        by_subj[subj] = [f]
    else:
        by_subj[subj].append(f)
        
preds = []
trues = []
for subj, files in tqdm(by_subj.items()):
    true_class = files[0].split('/')[-2]
    batch = [torch.load(x) for x in files]
    batch = torch.stack(batch)
    with torch.no_grad():
        _, pred = model(batch.to('cuda:0'))
        pred_class = pred.argmax(dim=1)
        pred_class_prob = torch.max(torch.softmax(pred, dim=1), dim=1) 
    trues.append(true_class)
    preds.append(statistics.mode(pred_class.tolist()))

print('Macro F1:', metrics.f1_score([int(x) for x in trues], preds, average='macro'))