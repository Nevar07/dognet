import argparse
import logging
import os

import numpy as np
import torch
from torch import distributed
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from backbones import get_model
from dataset import get_dataloader
from losses import CombinedMarginLoss,Contrastive
from lr_scheduler import PolyScheduler
from partial_fc import PartialFC, PartialFCAdamW
from utils.utils_callbacks import CallBackLogging, CallBackVerification
from utils.utils_config import get_config
from utils.utils_logging import AverageMeter, init_logging
from eval1 import arc_eval
import cv2

assert torch.__version__ >= "1.9.0", "In order to enjoy the features of the new torch, \
we have upgraded the torch to 1.9.0. torch before than 1.9.0 may not work in the future."

try:
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    distributed.init_process_group("nccl")
except KeyError:
    world_size = 1
    rank = 0
    distributed.init_process_group(
        backend="nccl",
        init_method="tcp://127.0.0.1:12584",
        rank=rank,
        world_size=world_size,
    )

def load_eval_data():
    img_data = {}
    for i in os.listdir('data/arc/images'):
        img = cv2.imread(os.path.join('data/arc/images',i))
        img = cv2.resize(img, (224, 224))
        # img = cv2.resize(img, (112, 112))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img).unsqueeze(0).float()
        img.div_(255).sub_(0.5).div_(0.5)
        img = img.cuda()

        img_data[i] = img

    return img_data
    

def main(args):
    seed = 2333
    seed = seed + rank
    torch.manual_seed(seed)
    np.random.seed(seed)

    torch.cuda.set_device(args.local_rank)
    cfg = get_config(args.config)

    os.makedirs(cfg.output, exist_ok=True)
    init_logging(rank, cfg.output)

    summary_writer = (
        SummaryWriter(log_dir=os.path.join(cfg.output, "tensorboard"))
        if rank == 0
        else None
    )
    train_loader = get_dataloader(
        cfg.rec, local_rank=args.local_rank, batch_size=cfg.batch_size, dali=cfg.dali)
    backbone = get_model(
        cfg.network, dropout=0.0, fp16=cfg.fp16, num_features=cfg.embedding_size
    ).cuda()
    backbone = torch.nn.parallel.DistributedDataParallel(
        module=backbone, broadcast_buffers=False, device_ids=[args.local_rank], bucket_cap_mb=16,
        find_unused_parameters=True)

    backbone.train()
    # FIXME using gradient checkpoint if there are some unused parameters will cause error
    backbone._set_static_graph()

    margin_loss = CombinedMarginLoss(
        30,
        cfg.margin_list[0],
        cfg.margin_list[1],
        cfg.margin_list[2],
        cfg.interclass_filtering_threshold
    )

    cri_con = Contrastive(2)
    

    if cfg.optimizer == "sgd":
        module_partial_fc = PartialFC(
            margin_loss, cfg.embedding_size, cfg.num_classes,
            cfg.sample_rate, cfg.fp16)
        module_partial_fc.train().cuda()
        # TODO the params of partial fc must be last in the params list
        opt = torch.optim.SGD(
            params=[{"params": backbone.parameters()}, {"params": module_partial_fc.parameters()}],
            lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay)

    elif cfg.optimizer == "adamw":
        module_partial_fc = PartialFCAdamW(
            margin_loss, cfg.embedding_size, cfg.num_classes,
            cfg.sample_rate, cfg.fp16)
        module_partial_fc.train().cuda()
        opt = torch.optim.AdamW(
            params=[{"params": backbone.parameters()}, {"params": module_partial_fc.parameters()}],
            lr=cfg.lr, weight_decay=cfg.weight_decay)
    else:
        raise

    opt_con = torch.optim.Adam(
            params=[{"params": backbone.parameters()}, {"params": module_partial_fc.parameters()}],
            lr=cfg.lr, betas = (0.5,0.999))

    cfg.total_batch_size = cfg.batch_size * world_size
    cfg.warmup_step = cfg.num_image // cfg.total_batch_size * cfg.warmup_epoch
    cfg.total_step = cfg.num_image // cfg.total_batch_size * cfg.num_epoch

    lr_scheduler = PolyScheduler(
        optimizer=opt,
        base_lr=cfg.lr,
        max_steps=cfg.total_step,
        warmup_steps=cfg.warmup_step,
        last_epoch=-1
    )

    lrs_con = PolyScheduler(
        optimizer=opt_con,
        base_lr=cfg.lr,
        max_steps=cfg.total_step,
        warmup_steps=cfg.warmup_step,
        last_epoch=-1
    )

    best_auc = 0
    eval_data = load_eval_data()
    start_epoch = 0
    global_step = 0
    if cfg.resume:
        dict_checkpoint = torch.load(os.path.join(cfg.output, f"checkpoint_gpu_{rank}.pt"))
        start_epoch = dict_checkpoint["epoch"]
        global_step = dict_checkpoint["global_step"]
        backbone.module.load_state_dict(dict_checkpoint["state_dict_backbone"])
        module_partial_fc.load_state_dict(dict_checkpoint["state_dict_softmax_fc"])
        opt.load_state_dict(dict_checkpoint["state_optimizer"])
        lr_scheduler.load_state_dict(dict_checkpoint["state_lr_scheduler"])
        del dict_checkpoint

    for key, value in cfg.items():
        num_space = 25 - len(key)
        logging.info(": " + key + " " * num_space + str(value))

    callback_verification = CallBackVerification(
        val_targets=cfg.val_targets, rec_prefix=cfg.rec, summary_writer=summary_writer
    )
    callback_logging = CallBackLogging(
        frequent=cfg.frequent,
        total_step=cfg.total_step,
        batch_size=cfg.batch_size,
        start_step = global_step,
        writer=summary_writer
    )

    loss_am = AverageMeter()
    amp = torch.cuda.amp.grad_scaler.GradScaler(growth_interval=100)

    for epoch in range(start_epoch, cfg.num_epoch):

        if isinstance(train_loader, DataLoader):
            train_loader.sampler.set_epoch(epoch)
        for _, (img, local_labels) in enumerate(train_loader):
            global_step += 1
            local_embeddings = backbone(img)
            loss_arc: torch.Tensor = module_partial_fc(local_embeddings, local_labels, opt)
            loss_con = cri_con(local_embeddings, local_labels)

            loss = loss_arc + loss_con

            if cfg.fp16:
                amp.scale(loss).backward()
                amp.unscale_(opt)
                amp.unscale_(opt_con)
                torch.nn.utils.clip_grad_norm_(backbone.parameters(), 5)
                amp.step(opt)
                amp.step(opt_con)
                amp.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(backbone.parameters(), 5)
                opt.step()
                opt_con.step()

            opt.zero_grad()
            opt_con.zero_grad()
            lr_scheduler.step()
            lrs_con.step()

            with torch.no_grad():
                loss_am.update(loss.item(), 1)
                callback_logging(global_step, loss_am, epoch, cfg.fp16, lr_scheduler.get_last_lr()[0], amp)

                if global_step % cfg.verbose == 0 and global_step > 0:
                    callback_verification(global_step, backbone)


        if cfg.save_all_states:
            checkpoint = {
                "epoch": epoch + 1,
                "global_step": global_step,
                "state_dict_backbone": backbone.module.state_dict(),
                "state_dict_softmax_fc": module_partial_fc.state_dict(),
                "state_optimizer": opt.state_dict(),
                "state_lr_scheduler": lr_scheduler.state_dict()
            }
            torch.save(checkpoint, os.path.join(cfg.output, f"checkpoint_gpu_{rank}.pt"))

        if rank == 0:
            path_module = os.path.join(cfg.output, "model.pt")
            torch.save(backbone.module.state_dict(), path_module)

        if cfg.dali:
            train_loader.reset()

        with torch.no_grad():
            if torch.cuda.device_count() > 1:
                torch.save(backbone.module.state_dict(), 'models/model.pt')
            else:
                torch.save(backbone.state_dict(), 'models/model.pt')

            # eval_loss,eval_auc = arc_eval(cfg.network,cfg.rec,eval_data)
            eval_loss, eval_auc = arc_eval(cfg.network, "./data/arc/train", eval_data)
            logging.info(f'eval loss: {eval_loss}, eval auc: {eval_auc}')

        if (eval_auc*5 - eval_loss) > best_auc:
            best_auc = (eval_auc*5 - eval_loss)
            if torch.cuda.device_count() > 1:
                torch.save(backbone.module.state_dict(), os.path.join(cfg.output,'best_model.pt'))
            else:
                torch.save(backbone.state_dict(),os.path.join(cfg.output, 'best_model.pt'))
            logging.info(f'best model saved at epoch {epoch}')

    if rank == 0:
        path_module = os.path.join(cfg.output, "model.pt")
        torch.save(backbone.module.state_dict(), path_module)

        # from torch2onnx import convert_onnx
        # convert_onnx(backbone.module.cpu().eval(), path_module, os.path.join(cfg.output, "model.onnx"))

    distributed.destroy_process_group()


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser(
        description="Distributed Arcface Training in Pytorch")
    parser.add_argument("--config", type=str,default='configs/dog.py', help="py config file")
    parser.add_argument("--local_rank", type=int, default=0, help="local_rank")
    main(parser.parse_args())
