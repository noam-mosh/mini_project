import argparse
import logging
import yaml
from transformers import AutoModelForObjectDetection, AutoImageProcessor
from torchmetrics.detection import MeanAveragePrecision, IntersectionOverUnion
from torch.utils.data import DataLoader
import torch
from pytorch_lightning import Trainer
import unittest
import os
import datasets
import pytorch_lightning as pl

# _logger = logging.getLogger('train')

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
    def __init__(self, lr, lr_backbone, weight_decay, id2label, label2id):
        super().__init__()
        # replace COCO classification head with custom head
        self.model = AutoModelForObjectDetection.from_pretrained("microsoft/conditional-detr-resnet-50",
                                                                 id2label=id2label,
                                                                 label2id=label2id,
                                                                 num_labels=7,   
                                                                 ignore_mismatched_sizes=True)
        # see https://github.com/PyTorchLightning/pytorch-lightning/pull/1896
        self.lr = lr
        self.lr_backbone = lr_backbone
        self.weight_decay = weight_decay
        # self.train_map = MeanAveragePrecision(box_format="xywh", class_metrics=False)
        self.validation_map = MeanAveragePrecision(box_format="xywh", class_metrics=True)
        # self.test_map = MeanAveragePrecision(box_format="xywh", class_metrics=True)
        # self.iou = IntersectionOverUnion(box_format="xywh", class_metrics=True, respect_labels=False)
        self.id2label = id2label
        self.label2id = label2id
        self.train_epoch = 0
        

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
        pred_boxes = outputs.pred_boxes
        logits = outputs.logits 

        return loss, loss_dict, pred_boxes, logits

    def training_step(self, batch, batch_idx):
        loss, loss_dict, pred_boxes, logits = self.common_step(batch, batch_idx)
        batch_size = logits.shape[0]
        # logs metrics for each training_step,
        # and the average across the epoch
        self.log("training_loss", loss, batch_size=batch_size)
        for k, v in loss_dict.items():
            self.log("train_" + k, v.item())
        
        return loss


    def validation_step(self, batch, batch_idx):
        loss, loss_dict, pred_boxes, logits = self.common_step(batch, batch_idx)
        batch_size = logits.shape[0]
        self.log("validation_loss", loss, batch_size=batch_size)
        for k, v in loss_dict.items():
            self.log("validation_" + k, v.item())

        prob = logits.sigmoid()    
        prob = prob.view(logits.shape[0], -1)
        k_value = min(300, prob.size(1))
        topk_values, topk_indexes = torch.topk(prob, k_value, dim=1)
        scores = topk_values
        topk_boxes = torch.div(topk_indexes, logits.shape[2], rounding_mode="floor")
        labels = topk_indexes % logits.shape[2]
        boxes = torch.gather(pred_boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))
        
        preds = []
        targets = []
        for i in range(batch_size):
            preds.append(
                dict(
                    boxes=boxes[i],
                    scores= scores[i],
                    labels=labels[i],
                )
            )
            targets.append(
                dict(
                    boxes=batch["labels"][i]["boxes"],
                    labels=batch["labels"][i]["class_labels"],
                )
            )
        self.validation_map.update(preds, targets) 
        # self.iou.update(preds, targets)
        
        return loss

    def on_validation_epoch_end(self):
        mAPs = {"valid_" + k: v for k, v in self.validation_map.compute().items()}
        mAPs_per_class = mAPs.pop("valid_map_per_class")
        mARs_per_class = mAPs.pop("valid_mar_100_per_class")
        vals_per_class = mAPs.pop("valid_classes") 
        self.log_dict(mAPs)
        self.log_dict(
            {
                f"valid_map_{label}": value.type(torch.float32)
                for label, value in zip(self.id2label.values(), mAPs_per_class) if label != 'other'
            },
        )
        self.log_dict(
            {
                f"valid_mar_100_{label}": value.type(torch.float32)
                for label, value in zip(self.id2label.values(), mARs_per_class) if label != 'other'
            },
        )
      
        self.validation_map.reset()
        # IOUs = {"valid_" + k: v for k, v in self.iou.compute().items()} 
        # self.log_dict(IOUs)
        # self.iou.reset()


             
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
    
    train_dataset = datasets.CocoDetection(root=dataset_path, annFile='annotations_train.json', feature_extractor=feature_extractor, train=True)
    val_dataset = datasets.CocoDetection(root=dataset_path, annFile='annotations_test.json', feature_extractor=feature_extractor, train=False)

    print("Number of training examples:", len(train_dataset))
    print("Number of validation examples:", len(val_dataset))

    train_dataloader = DataLoader(train_dataset, collate_fn=train_dataset.collate_fn, batch_size=2, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, collate_fn=val_dataset.collate_fn, batch_size=2, num_workers=4)

    cats = train_dataset.coco.cats
    id2label = {k: v['name'] for k, v in cats.items()}
    label2id = {v['name']: k for k,v in cats.items()}

    model = Detr(lr=1e-5, lr_backbone=1e-5, weight_decay=1e-4, id2label=id2label, label2id=label2id)
    trainer = Trainer(accelerator='gpu', devices=1, max_epochs=300, gradient_clip_val=0.1)
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

if __name__ == '__main__':
    main()


