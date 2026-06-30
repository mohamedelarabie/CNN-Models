from utils.utils import *
from utils.metrics import ClassificationMetricsLogger

def test_one_epoch(epoch, test_dataloader, model, criterion, logger_test, tb_logger, accelerator,args, save_test_results_path):
    model.eval()
    device = next(model.parameters()).device
    loss = AverageMeter()
    all_predictions = []
    all_targets = []
    all_probabilities = []

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            out_net = model(inputs)
            out_criterion = criterion(out_net, targets)
            probabilities = torch.softmax(out_net, dim=1)
            _, predicted = torch.max(probabilities, 1)

            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    loss.update(out_criterion.item())
    metrics_logger = ClassificationMetricsLogger(model_name = args.config, num_classes = args.dataset["num_classes"],  device=device, save_dir = save_test_results_path)
    metrics = metrics_logger.log_epoch_metrics(epoch=epoch, out=all_predictions, target=all_targets, probabilities=all_probabilities)

    if accelerator.is_main_process:
        logger_test.info(f"Test Epoch: {epoch} | Loss: {loss.avg:.4f} | Accuracy: {metrics['accuracy']:.4f} | Precision: {metrics['precision']:.4f} | Recall: {metrics['recall']:.4f} | F1 Score: {metrics['f1_score']:.4f}")
        if tb_logger is not None:
            tb_logger.add_scalar('Test/Loss', loss.avg, epoch)
            tb_logger.add_scalar('Test/Accuracy', metrics['accuracy'], epoch)
            tb_logger.add_scalar('Test/Precision', metrics['precision'], epoch)
            tb_logger.add_scalar('Test/Recall', metrics['recall'], epoch)
            tb_logger.add_scalar('Test/F1_Score', metrics['f1_score'], epoch)
        
    logger_test.info(
        f"Test epoch {epoch}: Average losses: "
        f"Loss: {loss.avg:.4f} | "
        f"accuracy: {metrics['accuracy']:.6f} | "
        f"precision: {metrics['precision']:.6f} | "
        f"recall: {metrics['recall']:.6f} | "
        f"F1 Score: {metrics['f1_score']:.6f}  "
    )   
    accelerator.wait_for_everyone()
    return loss.avg