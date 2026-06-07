#!/bin/bash
#SBATCH --partition=allcpu
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --chdir=/home/zhaiyufe/PycharmProjects/ML_Fitting_1D_GISAXS
#SBATCH --job-name=ml1d_build_allcpu
#SBATCH --output=./TrainSetBuild/build-allcpu-%N-%j.out
#SBATCH --error=./TrainSetBuild/build-allcpu-%N-%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=yufeng.zhai@desy.de
#SBATCH --requeue
#SBATCH --signal=B:TERM@120

bash TrainSetBuild/build_trainset_common.sh "$@"
