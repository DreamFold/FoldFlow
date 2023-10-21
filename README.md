<div align="center">

# SE(3)-Stochastic Flow Matching for Protein Backbone Generation

[![OT-CFM Preprint](http://img.shields.io/badge/paper-arxiv.2302.00482-B31B1B.svg)](https://arxiv.org/abs/2310.02391)
[![pytorch](https://img.shields.io/badge/PyTorch_1.13+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)

</div>

## Description
This is the official repository for the paper [SE(3) diffusion model with application to protein backbone generation](https://arxiv.org/abs/2310.02391). 

We propose a new family of [Flow Matching](https://openreview.net/forum?id=PqvMRDCJT9t) methods called FoldFlow tailored for distributions on SE(3) and with a focus on protein backbone generation. Our 3 proposed methods are:

- The first one is **FoldFlow-base**. Inspired by [Riemannian Flow Matching](https://arxiv.org/abs/2302.03660), we develop a Flow Matching approach to generate data living on SO(3) manifold.
- The second one is **FoldFlow-OT** which generalizes FoldFlow-base by drawing samples from a minibatch optimal transport coupling similarly to [OT-CFM](https://arxiv.org/abs/2302.00482).
- The third one is **FoldFlow-SFM**, a stochastic version of FoldFlow-OT.

Our experiments include: 
- Generation of synthetic SO(3) data.
- Protein backbone design.
- Equilibrium conformation generation.

> Note that our methods can be adapted for all applications where data live on the SO(3)/SE(3) manifold.

![foldflow](media/foldflow-sfm_protein.gif)

## Installation

Install dependencies
```bash
# clone project
git clone https://github.com/DreamFold/FoldFlow.git
cd FoldFlow

# [OPTIONAL] create conda environment
conda create -n foldflow python=3.9
conda activate foldflow

# install requirements
pip install -r requirements.txt

```

To run our jupyter notebooks, use the following commands after installing our package.
```bash
# install ipykernel
conda install -c anaconda ipykernel

# install conda env in jupyter notebooj
python -m ipykernel install --user --name=foldflow

# launch our notebooks with the foldflow kernel
```

## Current Code 
The current repository only contains toy experiments for learning an SO(3) multimodal density using all three FoldFlow models.

## Planned Updates
- [ ] Inference code for protein experiments
- [ ] Training code for protein experiments
- [ ] Equilibrium conformation generation  

## Citations
If this codebase is useful towards other research efforts please consider citing us.

```
@misc{bose2023se3stochastic,
      title={SE(3)-Stochastic Flow Matching for Protein Backbone Generation}, 
      author={Avishek Joey Bose and Tara Akhound-Sadegh and Kilian Fatras and Guillaume Huguet and Jarrid Rector-Brooks and Cheng-Hao Liu and Andrei Cristian Nica and Maksym Korablyov and Michael Bronstein and Alexander Tong},
      year={2023},
      eprint={2310.02391},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```


## Contribute

We welcome issues and pull requests (especially bug fixes) and contributions.
We will try our best to improve readability and answer questions!


## Licences
<p xmlns:cc="http://creativecommons.org/ns#" xmlns:dct="http://purl.org/dc/terms/"><a property="dct:title" rel="cc:attributionURL" href="https://github.com/Dreamfold/foldflow">FoldFlow</a> by <a rel="cc:attributionURL dct:creator" property="cc:attributionName" href="https://dreamfold.ai">Dreamfold</a> is licensed under <a href="http://creativecommons.org/licenses/by-nc/4.0/?ref=chooser-v1" target="_blank" rel="license noopener noreferrer" style="display:inline-block;">Attribution-NonCommercial 4.0 International<img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/cc.svg?ref=chooser-v1"><img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/by.svg?ref=chooser-v1"><img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/nc.svg?ref=chooser-v1"></a></p>

### Warning: the current code uses PyTorch 1.13 and torchdyn 1.0.6.

This code base is heavily inspired from the TorchCFM library! You can check Flow Matching with data living on Euclidean spaces there https://github.com/atong01/conditional-flow-matching

