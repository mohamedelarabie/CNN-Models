from torch.amp import autocast

def train_one_epoch(model, criterion, train_dataloader, optimizer, epoch, logger_train, tb_logger, current_step, accelerator):

    model.train()
    device = next(model.parameters()).device
    
    for batch_idx, (inputs, targets) in enumerate(train_dataloader):
        inputs, targets = inputs.to(device), targets.to(device)

        with accelerator.autocast():
            optimizer.zero_grad()
            out_net = model(inputs)
            out_criterion  = criterion(out_net, targets)
            accelerator.backward(out_criterion )
            optimizer.step()
            current_step += 1
            accelerator.wait_for_everyone()
        
        if current_step % 10 == 0 and accelerator.is_main_process:
            logger_train.info(f"Epoch [{epoch}] Step [{current_step}] Loss: {out_criterion.item():.4f}")
            if tb_logger is not None:
                tb_logger.add_scalar('train/loss', out_criterion.item(), current_step)
        
    return current_step