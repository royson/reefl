# Recurrent Early Exits for Federated Learning with Heterogeneous Clients

## Abstract
> Federated learning (FL) has enabled distributed learning of a model across multiple clients in a privacy-preserving manner. One of the main challenges of FL is to accommodate clients with varying hardware capacities; clients have differing compute and memory requirements. To tackle this challenge, recent state-of-the-art approaches leverage the use of early exits. Nonetheless, these approaches fall short of mitigating the challenges of joint learning multiple exit classifiers, often relying on hand-picked heuristic solutions for knowledge distillation among classifiers and/or utilizing additional layers for weaker classifiers. In this work, instead of utilizing multiple classifiers, we propose a recurrent early exit approach named ReeFL that fuses features from different sub-models into a single shared classifier. Specifically, we use a transformer-based early-exit module shared among sub-models to i) better exploit multi-layer feature representations for task-specific prediction and ii) modulate the feature representation of the backbone model for subsequent predictions. We additionally present a per-client self-distillation approach where the best sub-model is automatically selected as the teacher of the other sub-models at each client. Our experiments on standard image and speech classification benchmarks across various emerging federated fine-tuning baselines demonstrate ReeFL's effectiveness over previous works.

### Directory Structure \& Main Files

This codebase is built on top of [FedL2P: Federated Learning to Personalize (NeurIPS'23)](https://github.com/royson/fedl2p).

### Datasets

All datasets are to be placed in `data.args.path_to_data`. Data is partitioned in `data.args.dataset_fl_root`. By default, `data.args.path_to_data=/datasets/ReeFL`.

- Cifar100 - automatically downloads and partition data

- SpeechCommands - automatically downloads and partition data

- FEMNIST - Follow the data preparation pipeline in [Leaf](https://github.com/TalwalkarLab/leaf/blob/master/data/femnist/README.md). We ran our experiments with 381 clients `./preprocess.sh -s niid --iu 381 -k 0 -t sample --smplseed 1694599561 --spltseed 1694599561`.

---

## Usage & Examples

### General Usage

```
python main.py {path_to_yaml_file} # you can pass multiple yaml files and arguments. Later yaml/arguments will take precedence.
python main.py ./wandb/{wandb_local_run_folder}/files/user_config.yaml # resume a previous run
```

Set the maximum GPU memory allocated for each client by overwriting argument `vram`. 

### ReeFL Examples

By default, ReeFL uses DeiT as pretrained backbone, which weights are automatically downloaded. For experiments with VIM, 1) please download the pretrained model from [Vim](https://github.com/hustvl/Vim), 2) place the model in local directory `base_models/`, 3) add `configs/reeFL/models/vim.yaml` to your list of arguments and modify `models.net.args.base_model` to your filename in `base_models/*.pt`

```
python main.py configs/reeFL/dataset/{dataset}.yaml configs/reeFL/accumulator/{early exit}.yaml configs/reeFL/adapter/{peft}.yaml models.net.args.no_of_exits={number of exits} name={run_name} vram={GPU usage (MB) per client}
```

:heavy_exclamation_mark: add `name={run_name}` to name your experiments, otherwise a random string will be used.

#### CIFAR100 Examples

```
alpha=1.0 # LDA partitioning alpha
num_of_exits=4 # we ran 4 or 12 in our experiments
vram=10000 # 10GB total per client. Spawns 4 clients per GPU for a GPU with 40+GB
additional_commands='wandb_args.mode=disabled' # disable wandb
early_exit='reeFL' # for other methods, see configs/reeFL/accumulator
num_clients=100

# frozen backbone
python main.py configs/reeFL/dataset/cifar100.yaml configs/reeFL/accumulator/${early_exit}.yaml data.args.lda_alpha=\{${alpha}:${num_clients}\} vram=${vram} models.net.args.no_of_exits=${num_of_exits} ${additional_commands} 

# full fine-tuning
python main.py configs/reeFL/dataset/cifar100.yaml configs/reeFL/accumulator/${early_exit}.yaml data.args.lda_alpha=\{${alpha}:${num_clients}\} vram=${vram} models.net.args.no_of_exits=${num_of_exits} ${additional_commands} models.net.args.freeze_base_model=False

# peft
peft='lora' # for other pefts, see configs/reeFL/adapter
python main.py configs/reeFL/dataset/cifar100.yaml configs/reeFL/accumulator/${early_exit}.yaml configs/reeFL/adapter/${peft}.yaml data.args.lda_alpha=\{${alpha}:${num_clients}\} vram=${vram} models.net.args.no_of_exits=${num_of_exits} ${additional_commands}
```

#### FEMNIST Examples

```
num_of_exits=4 # we ran 4 or 12 in our experiments
vram=10000 # 10GB total per client. Spawns 4 clients per GPU for a GPU with 40+GB
additional_commands='wandb_args.mode=disabled' # disable wandb
early_exit='reeFL' # for other methods, see configs/reeFL/accumulator

# frozen backbone
python main.py configs/reeFL/dataset/femnist.yaml configs/reeFL/accumulator/${early_exit}.yaml vram=${vram} models.net.args.no_of_exits=${num_of_exits} ${additional_commands} 

# full fine-tuning
python main.py configs/reeFL/dataset/femnist.yaml configs/reeFL/accumulator/${early_exit}.yaml vram=${vram} models.net.args.no_of_exits=${num_of_exits} ${additional_commands} models.net.args.freeze_base_model=False

# peft
peft='lora' # for other pefts, see configs/reeFL/adapter
python main.py configs/reeFL/dataset/femnist.yaml configs/reeFL/accumulator/${early_exit}.yaml configs/reeFL/adapter/${peft}.yaml vram=${vram} models.net.args.no_of_exits=${num_of_exits} ${additional_commands}
```

#### SpeechCommandsv2 Examples

```
num_of_exits=4 # we ran 4 or 12 in our experiments
vram=10000 # 10GB total per client. Spawns 4 clients per GPU for a GPU with 40+GB
additional_commands='wandb_args.mode=disabled data.args.classes=all models.net.args.accumulator.args.num_classes=35' # disable wandb and use full SpeechCmds dataset
early_exit='reeFL' # for other methods, see configs/reeFL/accumulator

# frozen backbone
python main.py configs/reeFL/dataset/commands.yaml configs/reeFL/accumulator/${early_exit}.yaml vram=${vram} models.net.args.no_of_exits=${num_of_exits} ${additional_commands} 

# full fine-tuning
python main.py configs/reeFL/dataset/commands.yaml configs/reeFL/accumulator/${early_exit}.yaml vram=${vram} models.net.args.no_of_exits=${num_of_exits} ${additional_commands} models.net.args.freeze_base_model=False

# peft
peft='lora' # for other pefts, see configs/reeFL/adapter
python main.py configs/reeFL/dataset/commands.yaml configs/reeFL/accumulator/${early_exit}.yaml configs/reeFL/adapter/${peft}.yaml vram=${vram} models.net.args.no_of_exits=${num_of_exits} ${additional_commands}
```

#### Additional Commands for ScaleFL

For 4 exits, we use
```
models.net.args.blks_to_exit=[3,5,8,11] app.args.width_scaling=[0.4,0.55,0.75,1.0]
```

and for 12 exits, we use

```
app.args.width_scaling=\[0.25,0.3,0.35,0.4,0.48,0.55,0.61,0.69,0.75,0.84,0.92,1.\]
```

#### Other useful arguments for debugging:
```
wandb_args.entity={your entity} # to specify your own wandb entity
run=false # config check without running
configs/test.yaml # disables wandb for a quick test
server.strategy.args.min_fit_clients=1 # samples one client per FL round
app.run.num_rounds=1 # set number of rounds to 1 for quick testing of training and evaluation pipeline
app.args.save_log_file={filename} # saves all results (by exp name) in a pickle file (by default, all ReeFL experiments are saved to reefl_results.pkl)
```
