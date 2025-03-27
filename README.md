# PyTorch Transfer Learning

A tool to train a CNN for image classification of [hymentoptera](https://www.inaturalist.org/taxa/47201-Hymenoptera) (Wasps, Ants, and Bees) using transfer learning from the
pre-trained model ResNet18. It is based on the following PyTorch tutorial, [Transfer Learning for Computer Vision Tutorial](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html).

## Prerequisites

- Git
- Docker
- Python >= 3.12 (prefer using [asdf](https://asdf-vm.com/), [pyenv](https://github.com/pyenv/pyenv), or [uv managed versions](https://docs.astral.sh/uv/concepts/python-versions/) to system python)
- [uv](https://docs.astral.sh/uv/)

  ```console
  > curl -LsSf https://astral.sh/uv/install.sh | sh
  ```

- [bump-my-version](https://callowayproject.github.io/bump-my-version/) tool

  ```console
  > uv tool install bump-my-version
  ```

## Installation

```console
> git clone git@github.com:eriksf/pytorch-transfer-learning.git
> cd pytorch-transfer-learning
> uv venv --seed --python 3.12
> uv sync
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

To update the version, use the `bump-my-version` tool. Use the `show-bump` subcommand to show version bumps based on the current version:

```console
> bump-my-version show-bump
0.1.0 ── bump ─┬─ major ─ 1.0.0
               ├─ minor ─ 0.2.0
               ╰─ patch ─ 0.1.1
```

To test the version bump before updating anything and show what files will be changed, use the `--dry-run` option to the `bump <version>` subcommand:

```console
> bump-my-version bump patch --dry-run -v
Starting BumpVersion 1.1.1
Reading configuration
  Reading config file: /Users/erik/Devel/git/pytorch-transfer-learning/pyproject.toml
  Parsing current version '0.1.0'
  No setup hooks defined
  Attempting to increment part 'patch'
    Values are now: major=0, minor=1, patch=1
  New version will be '0.1.1'
Dry run active, won't touch any files.

File pytorch_transfer_learning/version.py: replace `{current_version}` with `{new_version}`
  Found '0\.1\.0' at line 1: 0.1.0
  Would change file pytorch_transfer_learning/version.py:
    *** before pytorch_transfer_learning/version.py
    --- after pytorch_transfer_learning/version.py
    ***************
    *** 1 ****
    ! __version__ = '0.1.0'
    --- 1 ----
    ! __version__ = '0.1.1'

Processing config file: /Users/erik/Devel/git/pytorch-transfer-learning/pyproject.toml
  Found '0\.1\.0' at line 1: 0.1.0
  Would change file /Users/erik/Devel/git/pytorch-transfer-learning/pyproject.toml:tool.bumpversion.current_version:
    *** before /Users/erik/Devel/git/pytorch-transfer-learning/pyproject.toml:tool.bumpversion.current_version
    --- after /Users/erik/Devel/git/pytorch-transfer-learning/pyproject.toml:tool.bumpversion.current_version
    ***************
    *** 1 ****
    ! 0.1.0
    --- 1 ----
    ! 0.1.1
  Found '0\.1\.0' at line 1: 0.1.0
  Would change file /Users/erik/Devel/git/pytorch-transfer-learning/pyproject.toml:project.version:
    *** before /Users/erik/Devel/git/pytorch-transfer-learning/pyproject.toml:project.version
    --- after /Users/erik/Devel/git/pytorch-transfer-learning/pyproject.toml:project.version
    ***************
    *** 1 ****
    ! 0.1.0
    --- 1 ----
    ! 0.1.1
No pre-commit hooks defined
  Would not commit
  Would not tag
No post-commit hooks defined
Done.
```

To do the actual update, use the `bump <version>` subcommand to update the version in both the `pyproject.toml` and the `pytorch_transfer_learning/version.py` files:


```console
> bump-my-version bump patch
```

After updating the version and committing the changes back to the repo, you should `tag` the repo to match this version:

```console
> git tag -a 0.1.1 -m "Version 0.1.1"
> git push origin 0.1.1
```
