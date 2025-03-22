"""
Transfer Learning for Computer Vision Tutorial
==============================================
**Author**: `Sasank Chilamkurthy <https://chsasank.github.io>`_

In this tutorial, you will learn how to train a convolutional neural network for
image classification using transfer learning. You can read more about the transfer
learning at `cs231n notes <https://cs231n.github.io/transfer-learning/>`__

Quoting these notes,

    In practice, very few people train an entire Convolutional Network
    from scratch (with random initialization), because it is relatively
    rare to have a dataset of sufficient size. Instead, it is common to
    pretrain a ConvNet on a very large dataset (e.g. ImageNet, which
    contains 1.2 million images with 1000 categories), and then use the
    ConvNet either as an initialization or a fixed feature extractor for
    the task of interest.

These two major transfer learning scenarios look as follows:

-  **Finetuning the ConvNet**: Instead of random initialization, we
   initialize the network with a pretrained network, like the one that is
   trained on imagenet 1000 dataset. Rest of the training looks as
   usual.
-  **ConvNet as fixed feature extractor**: Here, we will freeze the weights
   for all of the network except that of the final fully connected
   layer. This last fully connected layer is replaced with a new one
   with random weights and only this layer is trained.

"""
# License: BSD
# Author: Sasank Chilamkurthy
# Author: Erik Ferlanti

import logging
import os

import click
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision
from click_loglevel import LogLevel
from rich.console import Console
from rich.logging import RichHandler
from torch.optim import lr_scheduler
from torchvision import datasets, models

from .functions import data_transforms, image_show, save_model, train_model, visualize_model
from .version import __version__

cudnn.benchmark = True
console = Console()

LOG_FORMAT = '%(asctime)s [%(name)s.%(funcName)s - %(levelname)s] %(message)s'
LOG_FORMAT_RICH = '%(message)s'
rh = RichHandler(console=console)
rh.setFormatter(logging.Formatter(LOG_FORMAT_RICH))
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, handlers=[rh])
logger = logging.getLogger(__name__)


def set_logging_level(ctx, param, value):
    """
    Callback function for click that sets the logging level.
    """
    logging.getLogger('pytorch_transfer_learning').setLevel(value)
    return value


def set_log_file(ctx, param, value):
    """
    Callback function for click that sets a log file.
    """
    if value:
        fileHandler = logging.FileHandler(value, mode='w')
        logFormatter = logging.Formatter(LOG_FORMAT)
        fileHandler.setFormatter(logFormatter)
        logging.getLogger('pytorch_transfer_learning').addHandler(fileHandler)
    return value

@click.command()
@click.version_option(__version__)
@click.option('--log-level', type=LogLevel(), default=logging.INFO, is_eager=True, callback=set_logging_level, help='Set the log level', show_default=True)
@click.option('--log-file', type=click.Path(writable=True), is_eager=True, callback=set_log_file, help='Set the log file')
@click.option('--data-dir', type=click.Path(exists=True, readable=True, writable=True), default='hymenoptera_data', show_default=True, help='Set the data directory')
@click.option('--scenario', type=click.Choice(['finetuning', 'fixedfeature']), default='finetuning', help='Transfer learning scenario.', show_default=True)
@click.option('--model-dir', 'model_dir', type=click.Path(exists=True, readable=True, writable=True), default='.', help='Set the model directory', show_default=True)
@click.option('--output-dir', type=click.Path(exists=True, readable=True, writable=True), default='.', help='Set the output directory', show_default=True)
@click.option('--epochs', type=int, default=25, show_default=True, help='The number of epochs to train the model')
def main(log_level, log_file, data_dir, scenario, model_dir, output_dir, epochs):
    """Train a CNN for hymenoptera classification using transfer learning
    from the pre-trained model ResNet18.
    """
    console.print("[bold blue]Training a CNN for hymenoptera classification using transfer learning[/bold blue]")
    console.print(f"PyTorch version: [green]{torch.__version__}[/green]")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    console.print(f"Using device: [green]{device}[/green]")
    if device.type == 'cuda':
        console.print(f"CUDA version: [green]{torch.version.cuda}[/green]")
        console.print(f"GPU: [green]{torch.cuda.get_device_name(0)}[/green]")
        console.print(f"GPU Memory: [green]{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB[/green]")

    logger.debug(f"Logging is set to level {logging.getLevelName(log_level)}")
    if log_file:
        logger.debug(f"Log file is {log_file}")

    logger.debug(f"Using data directory: {data_dir}")
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4, shuffle=True, num_workers=4) for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    console.print(f"\nData directory: {data_dir}")
    console.print(f"Dataset sizes: {dataset_sizes}")
    class_names = image_datasets['train'].classes
    console.print(f"Classes: {class_names}")
    console.print(f"Transfer learning scenario: [green]{scenario}[/green]")

    logger.debug(f"Using output directory: {output_dir}")

    # Get a batch of training data
    inputs, classes = next(iter(dataloaders['train']))

    # Make a grid from batch
    out = torchvision.utils.make_grid(inputs)
    grid_file_path = os.path.join(output_dir, 'test_grid.png')
    image_show(out, grid_file_path, title=[class_names[x] for x in classes])
    console.print(f"\n[yellow]Example training data grid saved to [bold]'{grid_file_path}'[/bold][/yellow]")

    if scenario == "finetuning":
        # Finetuning the convnet
        model_ft = models.resnet18(weights='IMAGENET1K_V1')
        model_ft.name = 'resnet18-finetuned'
        num_ftrs = model_ft.fc.in_features
        # Here the size of each output sample is set to 2.
        # Alternatively, it can be generalized to ``nn.Linear(num_ftrs, len(class_names))``.
        model_ft.fc = nn.Linear(num_ftrs, 2)

        model_ft = model_ft.to(device)

        criterion = nn.CrossEntropyLoss()

        # Observe that all parameters are being optimized
        optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

        # Decay LR by a factor of 0.1 every 7 epochs
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

        model_ft = train_model(device, model_ft, dataloaders, dataset_sizes, criterion, optimizer_ft, exp_lr_scheduler, console, epochs)

        prediction_image_path = os.path.join(output_dir, f"{model_ft.name}_predictions.png")
        visualize_model(device, model_ft, dataloaders, class_names, prediction_image_path)
        console.print(f"\n[yellow]Prediction image for '{scenario}' model saved to [bold]'{prediction_image_path}'[/bold][/yellow]")

        model_path = save_model(model_ft, class_names, f'hymenoptera-{scenario}', model_dir)
        console.print(f"\n[yellow]Model saved to [bold]'{model_path}'[/bold][/yellow]")
    else:
        # ConvNet as fixed feature extractor
        model_conv = torchvision.models.resnet18(weights='IMAGENET1K_V1')
        model_conv.name = 'resnet18-fixedfeatures'
        for param in model_conv.parameters():
            param.requires_grad = False

        # Parameters of newly constructed modules have requires_grad=True by default
        num_ftrs = model_conv.fc.in_features
        model_conv.fc = nn.Linear(num_ftrs, 2)

        model_conv = model_conv.to(device)

        criterion = nn.CrossEntropyLoss()

        # Observe that only parameters of final layer are being optimized as
        # opposed to before.
        optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)

        # Decay LR by a factor of 0.1 every 7 epochs
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

        model_conv = train_model(device, model_conv, dataloaders, dataset_sizes, criterion, optimizer_conv, exp_lr_scheduler, console, epochs)

        prediction_image_path = os.path.join(output_dir, f"{model_conv.name}_predictions.png")
        visualize_model(device, model_conv, dataloaders, class_names, prediction_image_path)
        console.print(f"\n[yellow]Prediction image for '{scenario}' model saved to [bold]'{prediction_image_path}'[/bold][/yellow]")

        model_path = save_model(model_conv, class_names, f'hymenoptera-{scenario}', model_dir)
        console.print(f"\n[yellow]Model saved to [bold]'{model_path}'[/bold][/yellow]")

if __name__ == '__main__':
    main()
