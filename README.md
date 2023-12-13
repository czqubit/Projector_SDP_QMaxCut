# Projector-SDP-QMaxCut
Projector SDP Hierarchy for Quantum MaxCut: https://arxiv.org/abs/2307.15688

## Prerequisites
1. Python >=3.10, together with numpy and scipy; I use miniconda for installing and managing Python: https://docs.conda.io/projects/miniconda/en/latest/. 
2. Mosek Fusion API for Python: https://www.mosek.com/documentation/. Note that Mosek provides free academic license: https://www.mosek.com/products/academic-licenses/.

## Instructions
1. qmc_proj.py contains all the core functions. One example (ring-n) is included in the end, and to test it, simply do `python qmc_proj.py`.
2. qmc_proj_main.py shows one way to run qmc_proj tasks in batches, which is particularly useful when running tasks on clusters. 
