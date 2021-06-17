#$ -l tmem=8G
#$ -l gpu=true 
#$ -l h_rt=20:0:0
#$ -S /bin/bash
#$ -j y
#$ -wd /cluster/project2/Neuroblastoma/MedSegBaselines

~ /cluster/project2/Neuroblastoma/MedSegBaselines Run_test.py
