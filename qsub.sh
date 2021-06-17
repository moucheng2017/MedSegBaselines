#$ -l tmem=8G
#$ -l gpu=true 
#$ -l h_rt=20:0:0
#$ -S /bin/bash
#$ -j y
#$ -wd /cluster/project2/Neuroblastoma/MedSegBaselines

~/miniconda3/envs/env2021/bin/python Run_test.py
