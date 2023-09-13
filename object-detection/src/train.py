from .utils.torch_utils import ModelEMA, de_parallel, select_device, torch_distributed_zero_first
from .utils.metrics import fitness
from .utils.plots import plot_labels
from .utils.autoanchor import check_anchors
from .utils.loss import ComputeLoss
from .utils.loggers import Loggers
from .utils.general import (LOGGER,check_git_status, check_img_size, check_requirements,
                           check_suffix, check_yaml, colorstr, init_seeds,
                           intersect_dicts, labels_to_class_weights, methods,
                           strip_optimizer, labels_to_image_weights)
from utils.downloads import attempt_download
from custom.custom_datasets import create_dataloader
from utils.callbacks import Callbacks
from utils.autobatch import check_train_batch_size
from models.yolo import Model
from models.experimental import attempt_load
import val  # for end-of-epoch mAP
import os
import sys
import time
import random
import logging
from zipfile import ZipFile
from copy import deepcopy
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import SGD, lr_scheduler
# from tqdm.auto import tqdm

from coretex import ComputerVisionDataset, Experiment, folder_manager
from coretex import cache


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))


def getWeightsPath(weightsUrl: str) -> str:
    fileName = "yolov5n-7.pt"

    if not cache.exists(weightsUrl):
        cache.storeUrl(weightsUrl, fileName, True)

    cachePath = cache.getPath(weightsUrl)

    with ZipFile(cachePath, "r") as zipFile:
        zipFile.extractall(folder_manager.cache)

    return cachePath.with_suffix('.pt')


def train(experiment: Experiment[ComputerVisionDataset], hyp, callbacks):  # hyp is path/to/hyp.yaml or hyp dictionary
    logging.info(">> [Object Detection] Training started")

    val_num_epochs = experiment.parameters["epochs"]
    val_batch_size = experiment.parameters["batchSize"]
    weightDecay = experiment.parameters["weightDecay"]
    momentum = experiment.parameters["momentum"]
    learningRate = experiment.parameters["learningRate"]
    val_weights = getWeightsPath(experiment.parameters["weightsUrl"])
    val_img_size = experiment.parameters["imageSize"]  # DEFAULT IMAGE SIZE
    device = select_device('', batch_size=val_batch_size)

    # DDP mode
    if LOCAL_RANK != -1:
        logging.info(">> [Object Detection] DDP mode")

        assert torch.cuda.device_count() > LOCAL_RANK, 'insufficient CUDA devices for DDP command'
        torch.cuda.set_device(LOCAL_RANK)
        device = torch.device('cuda', LOCAL_RANK)
        dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "gloo")

    logging.info(">> [Object Detection] on_pretrain_routine_start")

    save_dir = Path('.')
    workers = 8
    data = 'data/custom.yaml'
    callbacks.run('on_pretrain_routine_start')

    # Directories
    logging.info(">> [Object Detection] Creating directories")

    w = save_dir / 'weights'  # weights dir
    w.mkdir(parents=True, exist_ok=True)  # make dir
    last, best = w / 'last.pt', w / 'best.pt'

    # Hyperparameters
    logging.debug(">> [Object Detection] Loading hyperparameters")

    if isinstance(hyp, str):
        with open(hyp, errors='ignore') as f:
            hyp = yaml.safe_load(f)  # load hyps dict
    logging.debug(colorstr('hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))

    # Save run settings
    logging.debug(">> [Object Detection] Saving run settings")

    with open(save_dir / 'hyp.yaml', 'w') as f:
        yaml.safe_dump(hyp, f, sort_keys=False)

    # Loggers
    # data_dict = None
    if RANK in [-1, 0]:
        loggers = Loggers(save_dir, val_weights, None, hyp, LOGGER)  # loggers instance
        # if loggers.wandb:
        #     data_dict = loggers.wandb.data_dict

        # Register actions
        for k in methods(loggers):
            callbacks.register_action(k, callback=getattr(loggers, k))

    # Config
    logging.debug(">> [Object Detection] Config")

    plots = True  # create plots
    cuda = device.type != 'cpu'
    init_seeds(1 + RANK)
    # with torch_distributed_zero_first(LOCAL_RANK):
        # data_dict = data_dict or check_dataset(data)  # check if None
    # train_path, val_path = data_dict['train'], data_dict['val']
    # nc = int(data_dict['nc'])  # number of classes
    # names = data_dict['names']  # class namessss
    nc = len(experiment.dataset.classes)
    if nc <= 0:
        raise Exception(f">> [Workplace Safety] Tried to run training on invalid ({nc}) number of classes")

    names = experiment.dataset.classes.labels
    assert len(names) == nc, f'{len(names)} names found for nc={nc} dataset'  # check
    # is_coco = isinstance(val_path, str) and val_path.endswith('coco/val2017.txt')  # COCO dataset
    is_coco = False

    # Model
    logging.debug(">> [Object Detection] Loading model weights")

    check_suffix(val_weights, '.pt')  # check weights
    with torch_distributed_zero_first(LOCAL_RANK):
        val_weights = attempt_download(val_weights)  # download if not found locally
    ckpt = torch.load(val_weights, map_location='cpu')  # load checkpoint to CPU to avoid CUDA memory leak
    model = Model(ckpt['model'].yaml, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # create
    exclude = ['anchor'] if (hyp.get('anchors')) else []  # exclude keys
    csd = ckpt['model'].float().state_dict()  # checkpoint state_dict as FP32
    csd = intersect_dicts(csd, model.state_dict(), exclude=exclude)  # intersect
    model.load_state_dict(csd, strict=False)  # load
    logging.debug(f'Transferred {len(csd)}/{len(model.state_dict())} items from {val_weights}')  # report

    # Image size
    logging.debug(">> [Object Detection] Checking image size")

    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    imgsz = check_img_size(val_img_size, gs, floor=gs * 2)  # verify imgsz is gs-multiple

    # Batch size
    if RANK == -1 and val_batch_size == -1:  # single-GPU only, estimate best batch size
        logging.debug(">> [Object Detection] Estimating batch size")

        batch_size = check_train_batch_size(model, imgsz)
        loggers.on_params_update({"batch_size": batch_size})

    # Optimizer
    logging.debug(">> [Object Detection] Initializing optimizer")

    nbs = 64  # nominal batch size
    accumulate = max(round(nbs / val_batch_size), 1)  # accumulate loss before optimizing
    weightDecay *= val_batch_size * accumulate / nbs  # scale weight_decay
    logging.info(f"Scaled weight_decay = {hyp['weight_decay']}")

    g = [], [], []  # optimizer parameter groups
    bn = tuple(v for k, v in nn.__dict__.items() if 'Norm' in k)  # normalization layers, i.e. BatchNorm2d()
    for v in model.modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):  # bias
            g[2].append(v.bias)
        if isinstance(v, bn):  # weight (no decay)
            g[1].append(v.weight)
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):  # weight (with decay)
            g[0].append(v.weight)

    optimizer = SGD(g[2], lr=learningRate, momentum=momentum, nesterov=True)
    optimizer.add_param_group({'params': g[0], 'weight_decay': weightDecay})  # add g0 with weight_decay
    optimizer.add_param_group({'params': g[1]})  # add g1 (BatchNorm2d weights)
    logging.debug(f"{colorstr('optimizer:')} {type(optimizer).__name__} with parameter groups "
                f"{len(g[1])} weight (no decay), {len(g[0])} weight, {len(g[2])} bias")
    del g

    def lf(x): return (1 - x / val_num_epochs) * (1.0 - hyp['lrf']) + hyp['lrf']  # linearZ
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)  # plot_lr_scheduler(optimizer, scheduler, epochs)

    # EMA
    logging.debug(">> [Object Detection] Initializing EMA")

    ema = ModelEMA(model) if RANK in [-1, 0] else None

    start_epoch, best_fitness = 0, 0.0

    # Optimizer
    if ckpt['optimizer'] is not None:
        logging.debug(">> [Object Detection] Loading optimizer from checkpoint")

        optimizer.load_state_dict(ckpt['optimizer'])
        best_fitness = ckpt['best_fitness']

    # EMA
    if ema and ckpt.get('ema'):
        logging.debug(">> [Object Detection] Loading EMA from checkpoint")

        ema.ema.load_state_dict(ckpt['ema'].float().state_dict())
        ema.updates = ckpt['updates']

    # Epochs
    start_epoch = ckpt['epoch'] + 1
    if val_num_epochs < start_epoch:
        logging.debug(
            f"{val_weights} has been trained for {ckpt['epoch']} epochs. Fine-tuning for {val_num_epochs} more epochs.")
        val_num_epochs += ckpt['epoch']  # finetune additional epochs

    del ckpt, csd

    # DP mode
    if cuda and RANK == -1 and torch.cuda.device_count() > 1:
        logging.debug(">> [Object Detection] Initializing DP mode")

        logging.warning('WARNING: DP not recommended, use torch.distributed.run for best DDP Multi-GPU results.\n'
                       'See Multi-GPU Tutorial at https://github.com/ultralytics/yolov5/issues/475 to get started.')
        model = torch.nn.DataParallel(model)

    # Trainloader
    logging.debug(">> [Object Detection] Initializing DataLoader for training")

    train_loader, dataset = create_dataloader(experiment,
                                              experiment.dataset,
                                              imgsz,
                                              val_batch_size // WORLD_SIZE,
                                              gs,
                                              hyp=hyp,
                                              augment=True,
                                              cache=None,
                                              rect=False,
                                              rank=LOCAL_RANK,
                                              workers=workers,
                                              prefix=colorstr('train: '),
                                              shuffle=True)
    mlc = int(np.concatenate(dataset.labels, 0)[:, 0].max())  # max label class
    nb = len(train_loader)  # number of batches
    assert mlc < nc, f'Label class {mlc} exceeds nc={nc}. Possible class labels are 0-{nc - 1}'

    # Process 0
    if RANK in [-1, 0]:
        logging.debug(">> [Object Detection] Initializing DataLoader for validation")

        val_loader = create_dataloader(experiment,
                                       experiment.dataset,
                                       imgsz,
                                       val_batch_size // WORLD_SIZE * 2,
                                       gs,
                                       hyp=hyp,
                                       cache=None,
                                       rect=True,
                                       rank=-1,
                                       workers=workers * 2,
                                       pad=0.5,
                                       prefix=colorstr('val: '))[0]

        labels = np.concatenate(dataset.labels, 0)
        if plots:
            plot_labels(labels, names, save_dir)
        check_anchors(dataset, model=model, thr=hyp['anchor_t'], imgsz=imgsz)
        model.half().float()  # pre-reduce anchor precision
        callbacks.run('on_pretrain_routine_end')

    # DDP mode
    if cuda and RANK != -1:
        logging.debug(">> [Object Detection] Initializing DDP mode")

        model = DDP(model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK)

    # Model attributes
    nl = de_parallel(model).model[-1].nl  # number of detection layers (to scale hyps)
    hyp['box'] *= 3 / nl  # scale to layers
    hyp['cls'] *= nc / 80 * 3 / nl  # scale to classes and layers
    hyp['obj'] *= (imgsz / 640) ** 2 * 3 / nl  # scale to image size and layers
    model.nc = nc  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc  # attach class weights
    model.names = names

    # Start training
    logging.debug(">> [Object Detection] Preparing to train")

    t0 = time.time()
    maps = np.zeros(nc)  # mAP per class
    nw = max(round(hyp['warmup_epochs'] * nb), 100)  # number of warmup iterations, max(3 epochs, 100 iterations)
    # nw = min(nw, (epochs - start_epoch) / 2 * nb)  # limit warmup to < 1/2 of training
    last_opt_step = -1
    results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
    f1 = 0
    scheduler.last_epoch = start_epoch - 1  # do not move
    scaler = amp.GradScaler(enabled=cuda)
    compute_loss = ComputeLoss(model)  # init loss class
    logging.debug(">> [Object Detection] on_train_start")
    callbacks.run('on_train_start')
    logging.debug(f'Image sizes {imgsz} train, {imgsz} val\n'
                f'Using {train_loader.num_workers * WORLD_SIZE} dataloader workers\n'
                f"Logging results to {colorstr('bold', save_dir)}\n"
                f'Starting training for {val_num_epochs} epochs...')

    endEpoch = val_num_epochs - start_epoch

    for epoch in range(start_epoch, val_num_epochs):  # epoch ------------------------------------------------------------------
        currentEpoch = epoch - start_epoch
        logging.info(f">> [Object Detection] Started epoch: {currentEpoch + 1}/{endEpoch}")

        logging.debug(">> [Object Detection] on_train_epoch_start")
        callbacks.run('on_train_epoch_start')
        model.train()

        cw = model.class_weights.cpu().numpy() * (1 - maps) ** 2 / nc  # class weights
        iw = labels_to_image_weights(dataset.labels, nc=nc, class_weights=cw)  # image weights
        dataset.indices = random.choices(range(dataset.n), weights=iw, k=dataset.n)  # rand weighted idx

        mloss = torch.zeros(3, device=device)  # mean losses
        if RANK != -1:
            train_loader.sampler.set_epoch(epoch)
        pbar = enumerate(train_loader)
        # if RANK in (-1, 0):
            # pbar = tqdm(pbar, total=nb, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')  # progress bar
        optimizer.zero_grad()
        for i, (imgs, targets, paths, _) in pbar:  # batch -------------------------------------------------------------
            logging.debug(f">> [Object Detection] Started batch {i + 1}/{len(train_loader)}")

            logging.debug(">> [Object Detection] on_train_batch_start")
            callbacks.run('on_train_batch_start')
            ni = i + nb * epoch  # number integrated batches (since train start)
            imgs = imgs.to(device, non_blocking=True).float() / 255  # uint8 to float32, 0-255 to 0.0-1.0

            # Warmup
            if ni <= nw:
                logging.debug(">> [Object Detection] Warmup")
                xi = [0, nw]  # x interp
                accumulate = max(1, np.interp(ni, xi, [1, nbs / val_batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    x['lr'] = np.interp(ni, xi, [hyp['warmup_bias_lr'] if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [hyp['warmup_momentum'], momentum])

            # Forward
            with amp.autocast(enabled=cuda):
                logging.debug(">> [Object Detection] Forward pass")
                pred = model(imgs)  # forward
                loss, loss_items = compute_loss(pred, targets.to(device))  # loss scaled by batch_size

                if RANK != -1:
                    loss *= WORLD_SIZE  # gradient averaged between devices in DDP mode

            # Backward
            logging.debug(">> [Object Detection] Backward pass")
            scaler.scale(loss).backward()

            # Optimize
            if ni - last_opt_step >= accumulate:
                logging.debug(">> [Object Detection] Optimizer")
                scaler.step(optimizer)  # optimizer.step
                scaler.update()
                optimizer.zero_grad()
                if ema:
                    ema.update(model)
                last_opt_step = ni

            # Log
            if RANK in (-1, 0):
                mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses

                mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
                # pbar.set_description(('%10s' * 2 + '%10.4g' * 5) %
                                    #  (f'{epoch}/{val_num_epochs - 1}', mem, *mloss, targets.shape[0], imgs.shape[-1]))
                logging.debug(('\n' + '%10s' * 7) % ('Epoch', 'gpu_mem', 'box', 'obj', 'cls', 'labels', 'img_size'))
                logging.debug(('%10s' * 2 + '%10.4g' * 5) %
                                     (f'{epoch}/{val_num_epochs - 1}', mem, *mloss, targets.shape[0], imgs.shape[-1]))

                logging.info(f"MLOSS: {mloss.cpu().detach().numpy()}, LOSS by batch size {loss.cpu().detach().numpy()[0]}")

                logging.debug(">> [Object Detection] on_train_batch_end")
                callbacks.run('on_train_batch_end', ni, model, imgs, targets, paths, plots, True)
                if callbacks.stop_training:
                    return

            logging.debug(f">> [Object Detection] Finished batch {i + 1}/{len(train_loader)}")
            # end batch ------------------------------------------------------------------------------------------------

        # Scheduler
        logging.debug(f">> [Object Detection] Scheduler")
        lr = [x['lr'] for x in optimizer.param_groups]  # for loggers
        scheduler.step()

        if RANK in (-1, 0):
            # mAP
            logging.debug(f">> [Object Detection] Updating mAP")
            logging.debug(f">> [Object Detection] on_train_epoch_end")
            callbacks.run('on_train_epoch_end', epoch=epoch)
            ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'names', 'stride', 'class_weights'])
            final_epoch = (epoch + 1 == val_num_epochs)
            if not final_epoch:  # Calculate mAP
                results, maps, _, f1 = val.run(
                    experiment,
                    experiment.dataset,
                    batch_size=val_batch_size // WORLD_SIZE * 2,
                    imgsz=imgsz,
                    model=ema.ema,
                    dataloader=val_loader,
                    save_dir=save_dir,
                    plots=False,
                    callbacks=callbacks,
                    compute_loss=compute_loss
                )

            if not experiment.submitMetrics({
                "loss": (epoch, float(loss.cpu().detach().numpy()[0])),
                "mAP@0.5": (epoch, float(results[2])),
                "mAP@0.5:0.95": (epoch, float(results[3]))
            }):
                logging.warning(">> [Workspace] Failed to submit metrics!")

            # Update best mAP

            fi = fitness(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
            if fi > best_fitness:
                best_fitness = fi
            log_vals = list(mloss) + list(results) + lr
            logging.debug(f">> [Object Detection] on_fit_epoch_end")
            callbacks.run('on_fit_epoch_end', log_vals, epoch, best_fitness, fi)

            # # Save model
            # if (final_epoch):  # if save
            ckpt = {
                'epoch': epoch,
                'best_fitness': best_fitness,
                'model': deepcopy(de_parallel(model)).half(),
                'ema': deepcopy(ema.ema).half(),
                'updates': ema.updates,
                'optimizer': optimizer.state_dict(),
                'wandb_id': loggers.wandb.wandb_run.id if loggers.wandb else None,
                'date': datetime.now().isoformat()}

            # Save last, best and delete
            logging.debug(f">> [Object Detection] Saving last model")
            torch.save(ckpt, last)
            if best_fitness == fi:
                logging.debug(f">> [Object Detection] Updating best model")
                torch.save(ckpt, best)
            # turned off saving model after every epoch because
            # it would be too big to upload to coretex
            # if (epoch > 0):
            #     torch.save(ckpt, w / f'epoch{epoch}.pt')
            del ckpt
            logging.debug(f">> [Object Detection] on_model_save")
            callbacks.run('on_model_save', last, epoch, final_epoch, best_fitness, fi)

        logging.debug(f">> [Object Detection] Finished epoch: {currentEpoch + 1}/{endEpoch}")

    if RANK in (-1, 0):
        logging.info(f'\n{epoch - start_epoch + 1} epochs completed in {(time.time() - t0) / 3600:.3f} hours.')
        for f in last, best:
            if f.exists():
                strip_optimizer(f)  # strip optimizers
                if f is best:
                    logging.info(f'\nValidating {f}...')
                    results, _, _, f1 = val.run(
                        experiment,
                        experiment.dataset,
                        batch_size=val_batch_size // WORLD_SIZE * 2,
                        imgsz=imgsz,
                        model=attempt_load(f, device).half(),
                        iou_thres=0.65 if is_coco else 0.60,  # best pycocotools results at 0.65
                        dataloader=val_loader,
                        save_dir=save_dir,
                        save_json=is_coco,
                        verbose=True,
                        plots=True,
                        callbacks=callbacks,
                        compute_loss=compute_loss)  # val best model with plots
                    if is_coco:
                        callbacks.run('on_fit_epoch_end', list(mloss) + list(results) + lr, epoch, best_fitness, fi)

        callbacks.run('on_train_end', last, best, plots, epoch, results)
        logging.info(f"Results saved to {colorstr('bold', save_dir)}")

    torch.cuda.empty_cache()
    return results, f1


def main(experiment: Experiment[ComputerVisionDataset], callbacks=Callbacks()) -> float:
    # Checks
    if RANK in (-1, 0):
        check_git_status()
        check_requirements(exclude=['thop'])

    hyp = ROOT / 'data/hyps/hyp.scratch-low.yaml'
    hyp = check_yaml(hyp)  # checks

    # Train
    _, f1 = train(experiment, hyp, callbacks)

    if WORLD_SIZE > 1 and RANK == 0:
        logging.info('Destroying process group... ')
        dist.destroy_process_group()

        with open(hyp, errors='ignore') as f:
            hyp = yaml.safe_load(f)  # load hyps dict
            if 'anchors' not in hyp:  # anchors commented in hyp.yaml
                hyp['anchors'] = 3

    return f1
