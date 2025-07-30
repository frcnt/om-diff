# OM-Diff: Inverse-design of organometallic catalysts with guided equivariant denoising diffusion

This repository is the official implementation of the [OM-Diff paper](https://doi.org/10.26434/chemrxiv-2024-882hh).

This repository builds on the [`lightning-hydra-template`](https://github.com/ashleve/lightning-hydra-template).
Explanations on e.g. how the config files are resolved can be found in the original template.

## Getting started

After creating an environment and installing the requirements, the repo can either be installed as a package or be used
without being installed by adding the path to root of the project to the `PYTHONPATH` environment variable.

### Setting the environment variables

A `.env` file has to be placed at the root of the project. It is used to contain environment variables that can be
accessed in the config files.

An example is provided in `.env.example`.

At this point, it is only used for defining:

* `PROJECT_ROOT`: pointing to the root of the project;
* `SCRATCH_PATH`: pointing to where the logs/checkpoints etc. should be dumped;
* `SCRATCH_COMPUTE_PATH`: pointing to where the dataset should be copied during training (e.g. a fast-access disk on a
  compute node).

## Usage

### How to train models?

There is a unique entry point for training the different models: `src/train.py`.

A training run can be launched using:
``
python src/train.py experiment=<experiment-name>
``
where `<experiment-name>`is the name of the experiment to be run.

The description of the experiment to be run should be placed in `config/experiment/<experiment-name>.yaml`.

A couple of experiment config files are provided and allow to:

* train a diffusion model;
* train a time-conditioned regressor;
* train a property predictor.

All experiments are by default run on the cross-coupling dataset investigated in the paper.

#### Train a diffusion model

```
python src/train.py experiment=train_diffusion_suzuki
```

#### Train a time-conditioned regressor

```
python src/train.py experiment=train_time_regressor_suzuki
```

#### Train a property predictor

```
python src/train.py experiment=train_regressor_suzuki
```

#### Train models using your own data

While a lot of the provided code/configs is overly specific to the experiments reported in the paper, training using
your own data should be straightforward.

Provided that you have saved the data in an `ase` database, you can easily interface to it using the
generic `ASEDataset` and `ASEDataModule`.

### How to sample from a trained diffusion model?

#### Unconditional sampling
This can be done using `src/sample_from_ckpt.py`.

#### Conditional sampling
This can be done using `src/cond_sample_from_ckpt.py`.


### How to evaluate the properties of complexes using a trained property predictor?
This can be done using `src/predict_on_samples.py`.
