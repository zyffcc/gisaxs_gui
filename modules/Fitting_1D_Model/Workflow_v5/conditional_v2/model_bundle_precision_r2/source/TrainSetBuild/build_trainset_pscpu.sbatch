#!/bin/bash
#SBATCH --partition=pscpu
#SBATCH --time=07:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --chdir=/home/zhaiyufe/PycharmProjects/ML_Fitting_1D_GISAXS
#SBATCH --job-name=ml1d_build_pscpu
#SBATCH --output=./TrainSetBuild/build-pscpu-%N-%j.out
#SBATCH --error=./TrainSetBuild/build-pscpu-%N-%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=yufeng.zhai@desy.de

bash TrainSetBuild/build_trainset_common.sh "$@"
