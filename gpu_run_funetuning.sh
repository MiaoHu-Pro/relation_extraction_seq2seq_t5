#!/bin/bash
#SBATCH --mail-user=miao.hu@soton.ac.uk
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#!SBATCH --mem=80G
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH -p swarm_a100
#SBATCH --gres=gpu:1
#SBATCH --time=60:00:00
#SBATCH --job-name=casrel
#SBATCH --output=./result_out/run_webnlg_t5_large_re_e50_bs16_lr_1e-4_keep_slash-%j.out

module add nvidia-cuda

nvidia-smi

echo "the job start "

# CUDA_VISIBLE_DEVICES="0,1" python -m torch.distributed.launch --nproc_per_node 2

#python3 run.py --dataset=nyt  --data_base_dir=./data   --output_dir=/scratch/mh1f25/save_model/ \
#--model_name_or_path=/scratch/mh1f25/bert-base-uncased --model_type bert  --do_train --do_test \
#--epochs 50  --batch_size 64  --learning_rate 5e-5
#

#python3 run.py --dataset=nyt  --data_base_dir=./data   --output_dir=/scratch/mh1f25/save_model/ \
#--model_name_or_path=/scratch/mh1f25/bert-large-uncased --model_type bert  --do_train --do_test \
#--epochs 100  --batch_size 64  --learning_rate 5e-5  --hidden_dim  1024
#

#python3 run.py --dataset=nyt  --data_base_dir=./data   --output_dir=/scratch/mh1f25/save_model/ \
#--model_name_or_path=/scratch/mh1f25/t5-large --model_type t5  --do_train --do_test \
#--epochs 50  --batch_size 16  --learning_rate 1e-5  --hidden_dim  1024

#python3 run.py --dataset=nyt  --data_base_dir=./data   --output_dir=/scratch/mh1f25/save_model/ \
#--model_name_or_path=/scratch/mh1f25/t5-base --model_type t5  --do_train --do_test \
#--epochs 50  --batch_size 16  --learning_rate 1e-4

#python3 run.py --dataset=nyt  --data_base_dir=./data   --output_dir=/scratch/mh1f25/save_model/ \
#--model_name_or_path=/scratch/mh1f25/gpt2 --model_type gpt2  --do_train --do_test \
#--epochs 20  --batch_size 64  --learning_rate 5e-5
##
#python3 run.py --dataset=nyt  --data_base_dir=./data   --output_dir=/scratch/mh1f25/save_model/ \
#--model_name_or_path=/scratch/mh1f25/bart-base --model_type bart  --do_train   --do_test \
#--epochs 20  --batch_size 64  --learning_rate 5e-5

#python3 run.py --dataset=webnlg  --data_base_dir=./data   --output_dir=/scratch/mh1f25/save_model/ \
#--model_name_or_path=../bert-base-uncased --model_type bert  --do_train --do_test \
#--epochs 3  --batch_size 128  --learning_rate 5e-5
#
#python3 run.py --dataset=webnlg  --data_base_dir=./data   --output_dir=/scratch/mh1f25/save_model/ \
#--model_name_or_path=../t5-base --model_type t5  --do_train --do_test \
#--epochs 3  --batch_size 128  --learning_rate 5e-5

#python3 run.py --dataset=webnlg  --data_base_dir=./data   --output_dir=/scratch/mh1f25/save_model/ \
#--model_name_or_path=/scratch/mh1f25/t5-base --model_type t5  --do_train --do_test \
#--epochs 50  --batch_size 16  --learning_rate 1e-4


python3 run.py --dataset=webnlg  --data_base_dir=./data   --output_dir=/scratch/mh1f25/save_model/ \
--model_name_or_path=/scratch/mh1f25/t5-large  --model_type t5  --do_train --do_test \
--epochs 50  --batch_size 16  --learning_rate 1e-4   --hidden_dim  1024


echo "the job end "




