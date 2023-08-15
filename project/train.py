import argparse
import logging
import yaml
import pytorch_lightning as pl
from transformers import AutoModelForObjectDetection, AutoImageProcessor
from torch.utils.data import DataLoader
import torch
import supervision
import transformers
from pytorch_lightning import Trainer
import unittest
import os
import datasets
from create_model import Detr

_logger = logging.getLogger('train')

# The first arg parser parses out only the --config argument, this argument is used to
# load a yaml file containing key-values that override the defaults for the main parser below
config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                    help='YAML config file specifying default arguments')


parser = argparse.ArgumentParser(description='Training conditional-DETR model for object detection task on TACO dataset')
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

def main():
    test = unittest.TestCase()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Using device:', device)

    dataset_path = '/datasets/TACO-master/data/'

    feature_extractor = AutoImageProcessor.from_pretrained("microsoft/conditional-detr-resnet-50")

    train_dataset = datasets.CocoDetection(img_folder=dataset_path, feature_extractor=feature_extractor, train=True)
    val_dataset = datasets.CocoDetection(img_folder=dataset_path, feature_extractor=feature_extractor, train=False)

    print("Number of training examples:", len(train_dataset))
    print("Number of test examples:", len(val_dataset))

    def collate_fn(batch):
        pixel_values = [item[0] for item in batch]
        # encoding = feature_extractor.pad_and_create_pixel_mask(pixel_values, return_tensors="pt")
        encoding = feature_extractor.pad(pixel_values, return_tensors="pt")
        labels = [item[1] for item in batch]
        batch = {}
        batch['pixel_values'] = encoding['pixel_values']
        batch['pixel_mask'] = encoding['pixel_mask']
        batch['labels'] = labels
        return batch

    train_dataloader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=2, shuffle=True, num_workers=2)
    val_dataloader = DataLoader(val_dataset, collate_fn=collate_fn, batch_size=2, num_workers=2)
    # batch = next(iter(train_dataloader))

    cats = train_dataset.coco.cats
    id2label = {k: v['name'] for k, v in cats.items()}
    id2label[8] = "unknown"

    model = Detr(lr=1e-4, lr_backbone=1e-5, weight_decay=1e-4, id2label=id2label)

    # outputs = model(pixel_values=batch['pixel_values'], pixel_mask=batch['pixel_mask'])

    MAX_EPOCHS = 8

    trainer = Trainer(accelerator='gpu', devices=1, max_epochs=MAX_EPOCHS, gradient_clip_val=0.1)
    trainer.fit(model)

    model = create_model(
        args.model,
        pretrained=args.pretrained,
        in_chans=in_chans,
        num_classes=args.num_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=args.drop_block,
        global_pool=args.gp,
        bn_momentum=args.bn_momentum,
        bn_eps=args.bn_eps,
        scriptable=args.torchscript,
        checkpoint_path=args.initial_checkpoint,
    )


    # Get data:
    train_dset = datasets.WSI_REGdataset(DataSet=args.dataset,
                                         tile_size=TILE_SIZE,
                                         target_kind=args.target,
                                         test_fold=args.test_fold,
                                         train=True,
                                         print_timing=args.time,
                                         transform_type=args.transform_type,
                                         n_tiles=args.n_patches_train,
                                         color_param=args.c_param,
                                         get_images=args.images,
                                         desired_slide_magnification=args.mag,
                                         DX=args.dx,
                                         loan=args.loan,
                                         er_eq_pr=args.er_eq_pr,
                                         slide_per_block=args.slide_per_block,
                                         balanced_dataset=args.balanced_dataset,
                                         RAM_saver=args.RAM_saver
                                         )
    test_dset = datasets.WSI_REGdataset(DataSet=args.dataset,
                                        tile_size=TILE_SIZE,
                                        target_kind=args.target,
                                        test_fold=args.test_fold,
                                        train=False,
                                        print_timing=False,
                                        transform_type='none',
                                        n_tiles=args.n_patches_test,
                                        get_images=args.images,
                                        desired_slide_magnification=args.mag,
                                        DX=args.dx,
                                        loan=args.loan,
                                        er_eq_pr=args.er_eq_pr,
                                        RAM_saver=args.RAM_saver
                                        )

    inf_dset = datasets.Infer_Dataset(DataSet=args.dataset,
                                      tile_size=TILE_SIZE,
                                      tiles_per_iter=args.tiles_per_iter,
                                      target_kind=args.target,
                                      # folds=args.folds,
                                      num_tiles=args.num_tiles,
                                      desired_slide_magnification=args.mag,
                                      dx=args.dx)
                                      # resume_slide=slide_num,
                                      # patch_dir=args.patch_dir,
                                      # chosen_seed=args.seed)

    NUM_SLIDES = len(inf_dset.image_file_names)
    sampler = None
    do_shuffle = True

    if args.supervised:
        temp_size = len(test_dset)
        train_dset, test_dset = random_split(test_dset, (int(temp_size*0.8), temp_size - int(temp_size*0.8)))

    if args.balanced_sampling:
        labels = pd.DataFrame(train_dset.dataset.target * train_dset.dataset.factor)
        n_pos = np.sum(labels == 'Positive').item()
        n_neg = np.sum(labels == 'Negative').item()
        weights = pd.DataFrame(np.zeros(n_pos+n_neg))
        # print("train size:" + str(len(train_dset)))
        # print("test size:" + str(len(test_dset)))
        # print("weights df shape is" + str(weights.shape))
        # print(weights[np.array(labels == 'Positive')])
        weights[np.array(labels == 'Positive')] = 1 / n_pos
        weights[np.array(labels == 'Negative')] = 1 / n_neg
        do_shuffle = False  # the sampler shuffles
        sampler = torch.utils.data.sampler.WeightedRandomSampler(weights=weights.squeeze(), num_samples=len(train_dset))
    loader_train = DataLoader(train_dset, batch_size=args.batch_size, shuffle=do_shuffle, num_workers=args.workers, pin_memory=True, sampler=sampler)

    inf_loader = DataLoader(inf_dset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    new_slide = True

    # loaders prints
    # print("train loader:")
    # print(loader_train)
    # print("train loader enumerate:")
    # print(list(enumerate(loader_train))[0])

    eval_workers = args.workers
    if args.distributed and ('tfds' in args.dataset or 'wds' in args.dataset):
        # FIXME reduces validation padding issues when using TFDS, WDS w/ workers and distributed training
        eval_workers = min(2, args.workers)
    if not args.extract_features:
        loader_eval = DataLoader(test_dset, batch_size=args.batch_size * 2, shuffle=False,
                             num_workers=args.workers, pin_memory=True)

    # setup mixup / cutmix
    collate_fn = None
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_args = dict(
            mixup_alpha=args.mixup,
            cutmix_alpha=args.cutmix,
            cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob,
            switch_prob=args.mixup_switch_prob,
            mode=args.mixup_mode,
            label_smoothing=args.smoothing,
            num_classes=args.num_classes
        )
        if args.prefetcher:
            assert not num_aug_splits  # collate conflict (need to support deinterleaving in collate mixup)
            collate_fn = FastCollateMixup(**mixup_args)
        else:
            mixup_fn = Mixup(**mixup_args)

    # wrap dataset in AugMix helper
    if num_aug_splits > 1:
        train_dset = AugMixDataset(train_dset, num_splits=num_aug_splits)

    # create data loaders w/ augmentation pipeiine
    # train_interpolation = args.train_interpolation
    # if args.no_aug or not train_interpolation:
    #     train_interpolation = data_config['interpolation']
    # timm_loader_train = create_loader(
    #     dataset_train,
    #     input_size=data_config['input_size'],
    #     batch_size=args.batch_size,
    #     is_training=True,
    #     use_prefetcher=args.prefetcher,
    #     no_aug=args.no_aug,
    #     re_prob=args.reprob,
    #     re_mode=args.remode,
    #     re_count=args.recount,
    #     re_split=args.resplit,
    #     scale=args.scale,
    #     ratio=args.ratio,
    #     hflip=args.hflip,
    #     vflip=args.vflip,
    #     color_jitter=args.color_jitter,
    #     auto_augment=args.aa,
    #     num_aug_repeats=args.aug_repeats,
    #     num_aug_splits=num_aug_splits,
    #     interpolation=train_interpolation,
    #     mean=data_config['mean'],
    #     std=data_config['std'],
    #     num_workers=args.workers,
    #     distributed=args.distributed,
    #     collate_fn=collate_fn,
    #     pin_memory=args.pin_mem,
    #     device=device,
    #     use_multi_epochs_loader=args.use_multi_epochs_loader,
    #     worker_seeding=args.worker_seeding,
    # )

    # loader_eval = create_loader(
    #     dataset_eval,
    #     input_size=data_config['input_size'],
    #     batch_size=args.validation_batch_size or args.batch_size,
    #     is_training=False,
    #     use_prefetcher=args.prefetcher,
    #     interpolation=data_config['interpolation'],
    #     mean=data_config['mean'],
    #     std=data_config['std'],
    #     num_workers=eval_workers,
    #     distributed=args.distributed,
    #     crop_pct=data_config['crop_pct'],
    #     pin_memory=args.pin_mem,
    #     device=device,
    # )

    # setup loss function
    if args.jsd_loss:
        assert num_aug_splits > 1  # JSD only valid with aug splits set
        train_loss_fn = JsdCrossEntropy(num_splits=num_aug_splits, smoothing=args.smoothing)
    elif mixup_active:
        # smoothing is handled with mixup target transform which outputs sparse, soft targets
        if args.bce_loss:
            train_loss_fn = BinaryCrossEntropy(target_threshold=args.bce_target_thresh)
        else:
            train_loss_fn = SoftTargetCrossEntropy()
    elif args.smoothing:
        if args.bce_loss:
            train_loss_fn = BinaryCrossEntropy(smoothing=args.smoothing, target_threshold=args.bce_target_thresh)
        else:
            train_loss_fn = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        train_loss_fn = nn.CrossEntropyLoss()
    train_loss_fn = train_loss_fn.to(device=device)
    validate_loss_fn = nn.CrossEntropyLoss().to(device=device)

    # setup checkpoint saver and eval metric tracking
    eval_metric = args.eval_metric
    best_metric = None
    best_epoch = None
    saver = None
    output_dir = None
    if utils.is_primary(args):
        if args.experiment:
            exp_name = args.experiment
            if args.subexperiment:
                subexp_name = args.subexperiment
        else:
            exp_name = '-'.join([
                datetime.now().strftime("%Y%m%d-%H%M%S"),
                safe_model_name(args.model),
                str(data_config['input_size'][-1])
            ])
        output_dir = utils.get_outdir(args.output if args.output else './output/train', exp_name, subexp_name)
        decreasing = True if eval_metric == 'loss' else False
        saver = utils.CheckpointSaver(
            model=model,
            optimizer=optimizer,
            args=args,
            model_ema=model_ema,
            amp_scaler=loss_scaler,
            checkpoint_dir=output_dir,
            recovery_dir=output_dir,
            decreasing=decreasing,
            max_history=args.checkpoint_hist
        )
        with open(os.path.join(output_dir, 'args.yaml'), 'w') as f:
            f.write(args_text)

    # setup learning rate schedule and starting epoch
    updates_per_epoch = len(loader_train)
    lr_scheduler, num_epochs = create_scheduler_v2(
        optimizer,
        **scheduler_kwargs(args),
        updates_per_epoch=updates_per_epoch,
    )
    start_epoch = 0
    if args.start_epoch is not None:
        # a specified start_epoch will always override the resume epoch
        start_epoch = args.start_epoch
    elif resume_epoch is not None:
        start_epoch = resume_epoch
    if lr_scheduler is not None and start_epoch > 0:
        if args.sched_on_updates:
            lr_scheduler.step_update(start_epoch * updates_per_epoch)
        else:
            lr_scheduler.step(start_epoch)

    if utils.is_primary(args):
        _logger.info(
            f'Scheduled epochs: {num_epochs}. LR stepped per {"epoch" if lr_scheduler.t_in_epochs else "update"}.')

    try:
        for epoch in range(start_epoch, num_epochs):
            if not args.extract_features:
                if hasattr(train_dset, 'set_epoch'):
                    train_dset.set_epoch(epoch)
                elif args.distributed and hasattr(loader_train.sampler, 'set_epoch'):
                    loader_train.sampler.set_epoch(epoch)
                train_metrics = train_one_epoch(
                    epoch,
                    model,
                    loader_train,
                    optimizer,
                    train_loss_fn,
                    args,
                    run,
                    lr_scheduler=lr_scheduler,
                    saver=saver,
                    output_dir=output_dir,
                    amp_autocast=amp_autocast,
                    loss_scaler=loss_scaler,
                    model_ema=model_ema,
                    mixup_fn=mixup_fn
                    )

            if args.distributed and args.dist_bn in ('broadcast', 'reduce'):
                if utils.is_primary(args):
                    _logger.info("Distributing BatchNorm running means and vars")
                utils.distribute_bn(model, args.world_size, args.dist_bn == 'reduce')
            inf_dset.reset_counter()
            eval_metrics = validate(
                model,
                inf_loader,
                train_loss_fn,
                # validate_loss_fn,
                args,
                run,
                amp_autocast=amp_autocast,
            )

            if model_ema is not None and not args.model_ema_force_cpu:
                if args.distributed and args.dist_bn in ('broadcast', 'reduce'):
                    utils.distribute_bn(model_ema, args.world_size, args.dist_bn == 'reduce')

                ema_eval_metrics = validate(
                    model_ema.module,
                    inf_loader,
                    validate_loss_fn,
                    args,
                    run,
                    amp_autocast=amp_autocast,
                    log_suffix=' (EMA)',
                )
                eval_metrics = ema_eval_metrics

            if output_dir is not None:
                lrs = [param_group['lr'] for param_group in optimizer.param_groups]
                utils.update_summary(
                    epoch,
                    train_metrics,
                    eval_metrics,
                    filename=os.path.join(output_dir, 'summary.csv'),
                    lr=sum(lrs) / len(lrs),
                    write_header=best_metric is None,
                    log_wandb=args.log_wandb and has_wandb,
                )

            if saver is not None:
                # save proper checkpoint with eval metric
                save_metric = eval_metrics[eval_metric]
                best_metric, best_epoch = saver.save_checkpoint(epoch, metric=save_metric)

            if lr_scheduler is not None:
                # step LR for next epoch
                lr_scheduler.step(epoch + 1, eval_metrics[eval_metric])

    except KeyboardInterrupt:
        pass

    if best_metric is not None:
        _logger.info('*** Best metric: {0} (epoch {1})'.format(best_metric, best_epoch))
