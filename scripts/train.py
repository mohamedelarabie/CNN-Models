import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path
import random
import logging
import math
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import ImageFile, Image
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from utils.logger import setup_logger
from utils.utils import save_checkpoint
from utils.optimizers import configure_optimizers
from utils.get_train_test_functions import get_train_test_functions
from utils.args import train_options
from utils.config import model_config
from datasets.usps_dataset import USPSDataset
from models.get_model import get_model
from utils.schedules import get_cosine_schedule_with_warmup
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs

def main():

    # Set up environment
    torch.backends.cudnn.benchmark = True
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    Image.MAX_IMAGE_PIXELS = None

    # Set up paths and configurations
    repo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    args = train_options()
    model_config_path = os.path.join(repo_path, 'configs', 'models', args.config + '.yaml')
    config = model_config(model_config_path)


    # Set random seeds
    seed = args.seed
    random.seed(seed)
    torch.manual_seed(seed)

    dp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

    accelerator = Accelerator(
        cpu=(args.device == "cpu"),
        mixed_precision=args.mixed_precision,  # 'no', 'fp16', 'bf16'
        gradient_accumulation_steps=1,
        kwargs_handlers=[dp_kwargs],
    )

    experiment_path = os.path.join(repo_path, 'experiments', args.config)
    save_test_results_path = os.path.join(repo_path, 'test_results')

    if accelerator.is_main_process:
        os.makedirs(experiment_path, exist_ok=True)
        os.makedirs(save_test_results_path, exist_ok=True)

    setup_logger('train', experiment_path, 'train_' + args.config, level=logging.INFO,
                        screen=True, tofile=True)
    setup_logger('test', experiment_path, 'test_' + args.config, level=logging.INFO,
                        screen=True, tofile=True)
    

    logger_train = logging.getLogger('train')
    logger_test = logging.getLogger('test')

    tb_logger = SummaryWriter(log_dir=experiment_path) if accelerator.is_main_process else None

    checkpoint_path = os.path.join(experiment_path, 'checkpoints')
    if accelerator.is_main_process:
        os.makedirs(checkpoint_path, exist_ok=True)

    train_dataset = USPSDataset(args.dataset['filename'], args.dataset['ImageSize'], mode='train')
    test_dataset = USPSDataset(args.dataset['filename'], args.dataset['ImageSize'], mode='test')

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=args.test_batch_size, num_workers=args.num_workers, shuffle=True)

    net, criterion = get_model(config)
    optimizer = configure_optimizers(net, args)
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=args.epochs//10, num_training_steps=args.epochs, base_lr=args.lr, end_lr=1e-5)

    net, criterion, optimizer, train_dataloader, test_dataloader, lr_scheduler = accelerator.prepare(
        net, criterion, optimizer, train_dataloader, test_dataloader, lr_scheduler
    )

    best_checkpoint = os.path.join(checkpoint_path,'checkpoint_best_loss.pth.tar')
    
    if os.path.exists(best_checkpoint):
        checkpoint = torch.load(best_checkpoint)
        net.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        best_loss = checkpoint['loss']
        current_step = start_epoch * math.ceil(len(train_dataloader.dataset) / args.batch_size)

        if accelerator.is_main_process:
            print(f"Loaded checkpoint from epoch {start_epoch} with loss {best_loss}")
        
    else:
        start_epoch = 0
        best_loss = 1e10
        current_step = 0
    if accelerator.is_main_process:
        logger_train.info(args)
        logger_train.info(config)
        logger_train.info(net)
        logger_train.info(optimizer)
    optimizer.param_groups[0]['lr'] = args.lr

    train_function, test_function = get_train_test_functions(config)
    for epoch in range(start_epoch, args.epochs):
        if accelerator.is_main_process:
            logger_train.info(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        current_step = train_function(net, criterion, train_dataloader, optimizer, epoch, logger_train, tb_logger, current_step, accelerator)
        loss = test_function(epoch, test_dataloader, net, criterion, logger_test, tb_logger, accelerator,args, save_test_results_path)
        lr_scheduler.step(epoch)
        is_best = loss < best_loss
        best_loss = min(loss, best_loss)
        if (epoch + 1) % args.save_every == 0:
            save_checkpoint(
                accelerator,
                {
                    "epoch": epoch + 1,
                    "state_dict": net.state_dict(),
                    "loss": loss,
                    "optimizer": optimizer.state_dict(),
                },
                is_best,
                os.path.join(checkpoint_path,"checkpoint_%03d.pth.tar" % (epoch + 1))
            )
            if is_best:
                logger_test.info('best checkpoint saved.')
            accelerator.wait_for_everyone()
if __name__ == "__main__":
    main()
