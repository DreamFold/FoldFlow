<div align="center">

# SE(3)-Stochastic Flow Matching for Protein Backbone Generation

[![OT-CFM Preprint](http://img.shields.io/badge/paper-arxiv.2310.02391-B31B1B.svg)](https://arxiv.org/abs/2310.02391)
[![pytorch](https://img.shields.io/badge/PyTorch_1.13+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)

</div>

**FoldFlow** is a [flow matching](https://github.com/atong01/conditional-flow-matching) generative model for protein design. FoldFlow works by generating protein structures as represented on the $SE(3)^N_0$ manifold. We investigate improvements such as including minibatch optimal transport conditional flows (**FoldFlow-OT**) which greatly improves designability and stochastic paths (**FoldFlow-SFM**), which increases the proportion of novel designs. For more information see our [arXiv](https://arxiv.org/abs/2310.02391) preprint.

This code heavily relies on and builds off of the [FrameDiff](https://github.com/jasonkyuyim/se3_diffusion) code. We thank the authors of that work for their efforts.

![foldflow](media/foldflow-sfm_protein.gif)

# Installation
To reproduce our results or train your own models you can install our codebase and its dependencies directly from this repository. The following command will clone our repository, create a conda environment from `se3.yml`, and install the dependencies. We tested the code with Python 3.9.15, and CUDA 11.6.1.


```bash
git clone https://github.com/DreamFold/FoldFlow.git
cd FoldFlow
conda env create -f se3.yml
conda activate se3
pip install -e .
```
# Inference

This project uses [hydra](https://hydra.cc) for configuration which allows easy command-line overrides and structured configs. You can find all the configurations files in `runner/config`.

In order to run inference with you own checkpoints or with our pretrained checkpoints, you need to specify the path to the checkpoint in the `runner/config/inference.yaml` file. During inference, we also evaluate FoldFlow designs using the Protein MPNN and ESMfold.

In `runner/config/inference.yaml` you can directly add the path to the checkpoints.

```yaml
inference:
  name: null
  gpu_id: 0  # CUDA GPU to use
  seed: 123
  full_ckpt_dir: None

  # Directory of software, weights, and outputs.
  pt_hub_dir: hub/checkpoints # ESMfold checkpoints
  pmpnn_dir: ./ProteinMPNN/
  output_dir: ./results/ # your output directory

  # Path to model weights.
  weights_path: path/to/ckpt/step_10.pth # Your FoldFlow checkpoint.
```
Once you have specified the path to the checkpoints, you can run inference using the following command:

```bash
python runner/inference.py
```
this will automatically use the configurations from `runner/config/inference.yaml`.

You can also modify the configurations from the command line. For example, if you want to change the path to the checkpoint and change the name of the experiment, you can run the following command:

```bash
python runner/inference.py inference.weights_path=path/to/new_ckpt.pth inference.name=new_ckpt
```

We followed the same inference procedure as [SE(3) diffusion model with application to protein backbone generation](https://github.com/jasonkyuyim/se3_diffusion). The results are saved in `results/` (or an another path that you specified), in the following way:

```bash
results/
    └── inference.name # Name of the experiment, if not specified it will be the time.
        └── length_50 # Length of the protein.
            ├── sample_0 # First FoldFlow design.
            │   ├── bb_traj_1.pdb # x_{t-1} diffusion trajectory.
            │   ├── sample_1.pdb # Sample at the final step.
            │   ├── x0_traj_1.pdb # x_0 model prediction trajectory
            │   ├── self_consistency # Self consistency results.
            │   │   ├── esmf # ESMFold predictions using ProteinMPNN sequences.
            │   │   │   ├── sample_0.pdb
            │   │   ├── parsed_pdbs.jsonl # Parsed chains for ProteinMPNN
            │   │   ├── sample_1.pdb
            │   │   ├── sc_results.csv # Self consistency summary metrics CSV
            │   │   └── seqs
            │           └── sample_1.fa # ProteinMPNN sequences
            └── sample_1
```

Note that saved models can be found [here](https://github.com/DreamFold/FoldFlow/releases/tag/0.1.0) for base, optimal transport (OT) and stochastic (SFM) foldflow models.

# Training

## Getting Started: Training FoldFlow on one protein
To get started and to make sure the code is working we recommend starting by training foldflow-base on a single protein. This should immediately work and produce the protein `2f60` in PDB.
```bash
python runner/train.py local=example
```
We expect this to converge in ~1500 steps and ~10-20 minutes on a V100. To train an OT model run:
```bash
python runner/train.py local=example flow_matcher.ot_plan=True
```
to train foldflow-sfm run:
```bash
python runner/train.py local=example flow_matcher.ot_plan=True flow_matcher.stochastic_paths=True
```

## Training on the Full Dataset
To get the full dataset, we supply two options:
1. We supply the full dataset in preprocessed form [here] TODO WHERE.
2. It can either be reprocessed from PDB using the steps described in the [se3_diffusion repository](https://github.com/jasonkyuyim/se3_diffusion#downloading-the-pdb-for-training).
We find (1) easier, but may become out of date as more PDBs are released.
<details>
  <summary>1. Downloading and unpacking our preprocessed data. </summary>

  We supply our `metadata.csv` file, which can be used to reproduce an identical training set in `data/metadata.csv`. Note that this file assumes all pickled data is located in `data/processed_pdbs/`, a new location requires rewriting this csv file.

  We also supply our saved data as tar file [here] TODO WHERE. Which can be extracted with
  ```bash
  tar xvzf processed_pdbs.tar.gz
  ```
  This may take a few minutes and requires ~32GB of disk space while unpacking.
</details>

<details>
    <summary>2. Downloading from PDB for training.</summary>

To get the training dataset, first download PDB then preprocess it with the provided scripts.
PDB can be downloaded from RCSB: https://www.wwpdb.org/ftp/pdb-ftp-sites#rcsbpdb.
Our scripts assume you download in **mmCIF format**.
Navigate down to "Download Protocols" and follow the instructions depending on your location.

> WARNING: Downloading PDB can take up to 1TB of space.

After downloading, you should have a directory formatted like this:
https://files.rcsb.org/pub/pdb/data/structures/divided/mmCIF/
```
00/
01/
02/
..
zz/
```
In this directory, unzip all the files:
```
gzip -d **/*.gz
```
Then run the following with <path_pdb_dir> replaced with the location of PDB.
```python
python process_pdb_dataset.py --mmcif_dir <pdb_dir>
```

See the script for more options. Each mmCIF will be written as a pickle file that
we read and process in the data loading pipeline. A `metadata.csv` will be saved
that contains the pickle path of each example as well as additional information
about each example for faster filtering.

Download the clusters at 30% sequence identity
at [rcsb](https://www.rcsb.org/docs/programmatic-access/file-download-services#sequence-clusters-data).
This download link also works at time of writing:
```
https://cdn.rcsb.org/resources/sequence/clusters/clusters-by-entity-30.txt
```
Place this file in `data/processed_pdb` or anywhere in your file system.
Update your config to point to the clustered data:
```yaml
data:
  cluster_path: ./data/processed_pdb/clusters-by-entity-30.txt
```

</details>

You can add the paths of to your data directly in `runner/config/data/default.yaml` or by adding your local configuration in `runner/config/local`. We suggest the latter, as it makes it easier to share your code with others. We provide an example of such configuration in `runner/config/local/example.yaml`.

## Evaluating Protein Models

Eval code coming soon!

## Toy SO(3) examples
Please find all the jupyter notebooks in `so3_experiments`, they are designed to be minimalistic and easy to follow and may be useful for other projects for applications of Flow Matching on SO(3).

To run our jupyter notebooks, use the following commands after installing our package.
```bash
# install ipykernel
conda install -c anaconda ipykernel

# install conda env in jupyter notebook
python -m ipykernel install --user --name=foldflow

# launch our notebooks with the foldflow kernel
```

### Third party source code

Our repo keeps a fork of [OpenFold](https://github.com/aqlaboratory/openfold) and [ProteinMPNN](https://github.com/dauparas/ProteinMPNN).
Each of these codebases are actively under development and you may want to refork.
Several files in `/data/` are adapted from [AlphaFold](https://github.com/deepmind/alphafold).


### Citation
If this codebase is useful towards other research efforts please consider citing us.

```
@inproceedings{bose2024se3stochastic,
      title={SE(3)-Stochastic Flow Matching for Protein Backbone Generation},
      author={Avishek Joey Bose and Tara Akhound-Sadegh and Guillaume Huguet and Killian Fatras and Jarrid Rector-Brooks and Cheng-Hao Liu and Andrei Cristian Nica and Maksym Korablyov and Michael Bronstein and Alexander Tong},
      year={2024},
      booktitle={The International Conference on Learning Representations (ICLR)},
}
```

### Contribute

We welcome issues and pull requests (especially bug fixes) and contributions.
We will try our best to improve readability and answer questions!

### Licences

<p xmlns:cc="http://creativecommons.org/ns#" xmlns:dct="http://purl.org/dc/terms/"><a property="dct:title" rel="cc:attributionURL" href="https://github.com/Dreamfold/foldflow">FoldFlow</a> by <a rel="cc:attributionURL dct:creator" property="cc:attributionName" href="https://dreamfold.ai">Dreamfold</a> is licensed under <a href="http://creativecommons.org/licenses/by-nc/4.0/?ref=chooser-v1" target="_blank" rel="license noopener noreferrer" style="display:inline-block;">Attribution-NonCommercial 4.0 International<img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/cc.svg?ref=chooser-v1"><img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/by.svg?ref=chooser-v1"><img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/nc.svg?ref=chooser-v1"></a></p>
