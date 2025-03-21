import logging
import os
import time
from tempfile import TemporaryDirectory

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

logger = logging.getLogger(__name__)

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
     'train': transforms.Compose([
         transforms.RandomResizedCrop(224),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
     ]),
     'val': transforms.Compose([
         transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
     ]),
}


def image_show(inp, filename=None, title=None):
    """Display image for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    if filename:
        plt.savefig(f"{filename}.png")
        plt.clf()


def train_model(device, model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, console, num_epochs=25):
    since = time.time()

    # Create a temporary directory to save training checkpoints
    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, 'best_model_params.pt')

        torch.save(model.state_dict(), best_model_params_path)
        logger.debug(f"Best model parameters will be saved to {best_model_params_path}")
        best_acc = 0.0

        with console.status("[magenta]Training model...") as status:
            for epoch in range(num_epochs):
                status.update(status=f'Training Epoch {epoch}')
                console.print(f'\nEpoch {epoch}/{num_epochs - 1}')
                console.print('-' * 10)

                # Each epoch has a training and validation phase
                for phase in ['train', 'val']:
                    if phase == 'train':
                        model.train()  # Set model to training mode
                    else:
                        model.eval()   # Set model to evaluate mode

                    running_loss = 0.0
                    running_corrects = 0

                    # Iterate over data.
                    for inputs, labels in dataloaders[phase]:
                        inputs = inputs.to(device)
                        labels = labels.to(device)

                        # zero the parameter gradients
                        optimizer.zero_grad()

                        # forward
                        # track history if only in train
                        with torch.set_grad_enabled(phase == 'train'):
                            outputs = model(inputs)
                            _, preds = torch.max(outputs, 1)
                            loss = criterion(outputs, labels)

                            # backward + optimize only if in training phase
                            if phase == 'train':
                                loss.backward()
                                optimizer.step()

                        # statistics
                        running_loss += loss.item() * inputs.size(0)
                        running_corrects += torch.sum(preds == labels.data)
                    if phase == 'train':
                        scheduler.step()

                    epoch_loss = running_loss / dataset_sizes[phase]
                    epoch_acc = running_corrects.double() / dataset_sizes[phase]

                    console.print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                    # deep copy the model
                    if phase == 'val' and epoch_acc > best_acc:
                        best_acc = epoch_acc
                        torch.save(model.state_dict(), best_model_params_path)
                        console.print("[yellow]Epoch accuracy is better than current best, saving model...[/yellow]")
                        logger.debug(f"Epoch accuracy is better than current best, model parameters will be saved to {best_model_params_path}")

                #console.print()

            time_elapsed = time.time() - since
            console.print('-' * 10)
            console.print(f'[green]Training complete in [bold]{time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s[/bold][/green]')
            console.print(f'Best val Acc: [green]{best_acc:4f}[/green]')

        # load best model weights
        model.load_state_dict(torch.load(best_model_params_path, weights_only=True))
    return model


def visualize_model(device, model, dataloaders, class_names, image_name, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title(f'predicted: {class_names[preds[j]]}')
                image_show(inputs.cpu().data[j])

                if images_so_far == num_images:
                    image_show(inputs.cpu().data[j], image_name)
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)


def visualize_model_predictions(device, model, class_names, img_path):
    was_training = model.training
    model.eval()

    img = Image.open(img_path)
    img = data_transforms['val'](img)
    img = img.unsqueeze(0)
    img = img.to(device)

    with torch.no_grad():
        outputs = model(img)
        _, preds = torch.max(outputs, 1)

        ax = plt.subplot(2,2,1)
        ax.axis('off')
        ax.set_title(f'Predicted: {class_names[preds[0]]}')
        image_show(img.cpu().data[0], f"{model.name}_custom_prediction")

        model.train(mode=was_training)
