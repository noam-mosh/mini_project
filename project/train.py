import argparse
import logging
import yaml
from transformers import AutoModelForObjectDetection, AutoImageProcessor
from torch.utils.data import DataLoader
import torch
from pytorch_lightning import Trainer
import unittest
import os
import datasets
import pytorch_lightning as pl

_logger = logging.getLogger('train')

# The first arg parser parses out only the --config argument, this argument is used to
# load a yaml file containing key-values that override the defaults for the main parser below
config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                    help='YAML config file specifying default arguments')

parser.add_argument('--sched', type=str, default='cosine', metavar='SCHEDULER',
                    help='LR scheduler (default: "step"')
parser.add_argument('--sched-on-updates', action='store_true', default=False,
                    help='Apply LR scheduler step on update instead of epoch end.')
parser.add_argument('--lr', type=float, default=None, metavar='LR',
                    help='learning rate, overrides lr-base if set (default: None)')
parser.add_argument('--lr-base', type=float, default=0.1, metavar='LR',
                    help='base learning rate: lr = lr_base * global_batch_size / base_size')
parser.add_argument('--lr-base-size', type=int, default=256, metavar='DIV',
                    help='base learning rate batch size (divisor, default: 256).')
parser.add_argument('--lr-base-scale', type=str, default='', metavar='SCALE',
                    help='base learning rate vs batch_size scaling ("linear", "sqrt", based on opt if empty)')
parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                    help='learning rate noise on/off epoch percentages')
parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                    help='learning rate noise limit percent (default: 0.67)')
parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                    help='learning rate noise std-dev (default: 1.0)')
parser.add_argument('--lr-cycle-mul', type=float, default=1.0, metavar='MULT',
                    help='learning rate cycle len multiplier (default: 1.0)')
parser.add_argument('--lr-cycle-decay', type=float, default=0.5, metavar='MULT',
                    help='amount to decay each learning rate cycle (default: 0.5)')
parser.add_argument('--lr-cycle-limit', type=int, default=1, metavar='N',
                    help='learning rate cycle limit, cycles enabled if > 1')
parser.add_argument('--lr-k-decay', type=float, default=1.0,
                    help='learning rate k-decay for cosine/poly (default: 1.0)')
parser.add_argument('--warmup-lr', type=float, default=1e-5, metavar='LR',
                    help='warmup learning rate (default: 1e-5)')
parser.add_argument('--min-lr', type=float, default=0, metavar='LR',
                    help='lower lr bound for cyclic schedulers that hit 0 (default: 0)')
parser.add_argument('--epochs', type=int, default=300, metavar='N',
                    help='number of epochs to train (default: 300)')
parser.add_argument('--epoch-repeats', type=float, default=0., metavar='N',
                    help='epoch repeat multiplier (number of times to repeat dataset epoch per train epoch).')
parser.add_argument('--start-epoch', default=None, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--decay-milestones', default=[90, 180, 270], type=int, nargs='+', metavar="MILESTONES",
                    help='list of decay epoch indices for multistep lr. must be increasing')
parser.add_argument('--decay-epochs', type=float, default=90, metavar='N',
                    help='epoch interval to decay LR')
parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                    help='epochs to warmup LR, if scheduler supports')
parser.add_argument('--warmup-prefix', action='store_true', default=False,
                    help='Exclude warmup period from decay schedule.'),
parser.add_argument('--cooldown-epochs', type=int, default=0, metavar='N',
                    help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                    help='patience epochs for Plateau LR scheduler (default: 10)')
parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                    help='LR decay rate (default: 0.1)')

parser = argparse.ArgumentParser(description='Training conditional-DETR model for Object Detection task on TACO dataset')
def _parse_args():
    # Do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)

    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text

class Detr(pl.LightningModule):

    def __init__(self, lr, lr_backbone, weight_decay, id2label):
        super().__init__()
        # replace COCO classification head with custom head
        self.model = AutoModelForObjectDetection.from_pretrained("microsoft/conditional-detr-resnet-50",
                                                                 id2label=id2label,
                                                                 ignore_mismatched_sizes=True)
        # see https://github.com/PyTorchLightning/pytorch-lightning/pull/1896
        self.lr = lr
        self.lr_backbone = lr_backbone
        self.weight_decay = weight_decay

    def forward(self, pixel_values, pixel_mask):
        outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)

        return outputs

    def common_step(self, batch, batch_idx):
        pixel_values = batch["pixel_values"]
        pixel_mask = batch["pixel_mask"]
        labels = [{k: v.to(self.device) for k, v in t.items()} for t in batch["labels"]]

        outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)

        loss = outputs.loss
        loss_dict = outputs.loss_dict

        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)
        # logs metrics for each training_step,
        # and the average across the epoch
        self.log("training_loss", loss)
        for k, v in loss_dict.items():
            self.log("train_" + k, v.item())

        return loss

    def validation_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)
        self.log("validation_loss", loss)
        for k, v in loss_dict.items():
            self.log("validation_" + k, v.item())

        return loss

    def configure_optimizers(self):
        param_dicts = [
            {"params": [p for n, p in self.named_parameters() if "backbone" not in n and p.requires_grad]},
            {
                "params": [p for n, p in self.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": self.lr_backbone,
            },
        ]
        optimizer = torch.optim.AdamW(param_dicts, lr=self.lr,
                                      weight_decay=self.weight_decay)

        return optimizer


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    
    args, args_text = _parse_args()

    dataset_path = '/datasets/TACO-master/data/'

    feature_extractor = AutoImageProcessor.from_pretrained("microsoft/conditional-detr-resnet-50")

    train_dataset = datasets.CocoDetection(img_folder=dataset_path, feature_extractor=feature_extractor, train=True)
    val_dataset = datasets.CocoDetection(img_folder=dataset_path, feature_extractor=feature_extractor, train=False)

    print("Number of training examples:", len(train_dataset))
    print("Number of test examples:", len(val_dataset))

    train_dataloader = DataLoader(train_dataset, collate_fn=train_dataset.collate_fn, batch_size=2, shuffle=True, num_workers=2)
    val_dataloader = DataLoader(val_dataset, collate_fn=train_dataset.collate_fn, batch_size=2, num_workers=2)

    cats = train_dataset.coco.cats
    id2label = {k: v['name'] for k, v in cats.items()}
    id2label[8] = "unknown"

    model = Detr(lr=1e-4, lr_backbone=1e-5, weight_decay=1e-4, id2label=id2label)

    trainer = Trainer(accelerator='gpu', devices=1, max_epochs=args.epochs, gradient_clip_val=0.1)
    trainer.fit(model, train_dataloaders=train_dataloader)
