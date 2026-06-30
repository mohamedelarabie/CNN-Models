import torch
import torch.optim as optim


def configure_optimizers(net, args):
    """
    Configure a single Adam optimizer for all trainable parameters.
    """

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, net.parameters()),
        lr=args.lr,
    )

    return optimizer