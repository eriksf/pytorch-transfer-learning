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

import click
import torch
import torch.backends.cudnn as cudnn
from click_loglevel import LogLevel
from rich.console import Console
from rich.logging import RichHandler

from .functions import load_model, visualize_model_predictions
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
@click.option('--model', 'model_path', type=click.Path(exists=True, readable=True), required=True, help='Set the model')
@click.option('--output-dir', type=click.Path(exists=True, readable=True, writable=True), default='.', help='Set the output directory', show_default=True)
@click.argument('image', type=click.Path(exists=True, readable=True), required=True)
def main(log_level, log_file, model_path, output_dir, image):
    """Predict the class of a given image based on the CNN model trained by transfer learning
    for hymenoptera classification.
    """
    console.print("[bold blue]Predict the class of an image based on the CNN trained for hymenoptera classification[/bold blue]")
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

    logger.debug(f"Using model: {model_path}")

    model, class_names = load_model(model_path)

    predicted_class, predicted_image = visualize_model_predictions(device, model, class_names, image, output_dir)
    console.print(f'\nPredicted class: [green]{predicted_class}[/green]')
    console.print(f"\n[yellow]Predicted image saved to [bold]'{predicted_image}'[/bold][/yellow]")

if __name__ == '__main__':
    main()
