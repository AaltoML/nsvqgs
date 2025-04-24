#!/bin/env bash
# * basic settings
source activate nsvqgs
hostname
nvidia-smi

# * data loading, path preparation
path_base=../../../data/GS
output_base=results/test
scene=truck
dataset=tandt
path_source="$path_base"/"$dataset"/"$scene"
path_output="$output_base"/"$dataset"/"$scene"

# * scripts to run GS model from scratch
CUDA_VISIBLE_DEVICES=0 python train_newvq.py  --port 4060 --vq_ncls 4096 --vq_ncls_sh 1024 \
  --vq_ncls_dc 1024 --vq_start_iter 20000 --quant_params sh dc rot scale --opacity_reg \
  --lambda_reg 1e-7 --max_prune_iter 20000 --eval -s=${path_source} -m=${path_output} \
   --total_iterations 45000 --fine_tune

# * rendering
python render.py -m ${path_output} --iteration 45000 --skip_train --load_quant

# * metrics calculation
python metrics.py -m ${path_output}