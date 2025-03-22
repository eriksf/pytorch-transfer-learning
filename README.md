# PyTorch Transfer Learning

A tool to train a CNN for image classification of [hymentoptera](https://www.inaturalist.org/taxa/47201-Hymenoptera) (Wasps, Ants, and Bees) using transfer learning from the
pre-trained model ResNet18. It is based on the following PyTorch tutorial, [Transfer Learning for Computer Vision Tutorial](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html).

## Prerequisites

- Git
- Docker
- Python >= 3.12 (prefer using [asdf](https://asdf-vm.com/) or [pyenv](https://github.com/pyenv/pyenv) to system python)
- [Poetry](https://python-poetry.org/) (prefer [asdf-poetry](https://github.com/asdf-community/asdf-poetry) plugin or installing with [pipx](https://github.com/pypa/pipx))

  ```console
  > curl -sSL https://install.python-poetry.org | python3 -
  ```

- poetry-bumpversion plugin

  ```console
  > poetry self add poetry-bumpversion
  ```

- poetry-plugin-export (if using Poetry >= 2.0)

  ```console
  > poetry self add poetry-plugin-export
  ```

- poetry-plugin-shell (if using Poetry >= 2.0)

  ```console
  > poetry self add poetry-plugin-shell
  ```

## Installation

```console
> git clone git@github.com:eriksf/pytorch-transfer-learning.git
> cd pytorch-transfer-learning
> poetry install
```

## Usage

```console
> train --help
Usage: train [OPTIONS]

  Train a CNN for hymenoptera classification using transfer learning from the
  pre-trained model ResNet18.

Options:
  --version                       Show the version and exit.
  --log-level [NOTSET|DEBUG|INFO|WARNING|ERROR|CRITICAL]
                                  Set the log level  [default: 20]
  --log-file PATH                 Set the log file
  --data-dir PATH                 Set the data directory  [default:
                                  hymenoptera_data]
  --scenario [finetuning|fixedfeature]
                                  Transfer learning scenario.  [default:
                                  finetuning]
  --model-dir PATH                Set the model directory  [default: .]
  --output-dir PATH               Set the output directory  [default: .]
  --epochs INTEGER                The number of epochs to train the model
                                  [default: 25]
  --help                          Show this message and exit.
```

```console
> predict --help
Usage: predict [OPTIONS] IMAGE

  Predict the class of a given image based on the CNN model trained by
  transfer learning for hymenoptera classification.

Options:
  --version                       Show the version and exit.
  --log-level [NOTSET|DEBUG|INFO|WARNING|ERROR|CRITICAL]
                                  Set the log level  [default: 20]
  --log-file PATH                 Set the log file
  --model PATH                    Set the model  [required]
  --output-dir PATH               Set the output directory  [default: .]
  --help                          Show this message and exit.
```

## Development

To update the version, use the `poetry version <major|minor|patch>` command (aided by the poetry-bumpversion plugin):

```console
> poetry version patch
Bumping version from 0.1.0 to 0.1.1
poetry_bumpversion: processed file pytorch_transfer_learning/version.py
```

This will update the version in both the `pyproject.toml` and the `pytorch_transfer_learning/version.py` files. If you want to test the version bump before updating files, you can use the `--dry-run` option:

```console
> poetry version patch --dry-run
Bumping version from 0.1.0 to 0.1.1
poetry_bumpversion: processed file pytorch_transfer_learning/version.py
```

After updating the version and committing the changes back to the repo, you should `tag` the repo to match this version:

```console
> git tag -a 0.1.1 -m "Version 0.1.1"
> git push origin 0.1.1
```
