#!/bin/bash
#SBATCH --partition=pscpu
#SBATCH --time=07:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --chdir=/home/zhaiyufe/PycharmProjects/ML_Fitting_1D_GISAXS
#SBATCH --job-name=ml1d_build_trainset
#SBATCH --output=./TrainSetBuild/build-%N-%j.out
#SBATCH --error=./TrainSetBuild/build-%N-%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=yufeng.zhai@desy.de

bash TrainSetBuild/build_trainset_common.sh "$@"

# Compatibility wrapper. Prefer:
# sbatch TrainSetBuild/build_trainset_pscpu.sbatch 100000 1000 42
# sbatch TrainSetBuild/build_trainset_allcpu.sbatch 100000 1000 42
#
# Four parallel jobs as a Slurm array on pscpu:
# sbatch --array=0-3 TrainSetBuild/build_trainset_pscpu.sbatch 100000 1000 42 4
#
# Four parallel jobs as a Slurm array on allcpu:
# sbatch --array=0-3 TrainSetBuild/build_trainset_allcpu.sbatch 100000 1000 42 4
#
# Single-component dataset, four-way pscpu array:
# sbatch --array=0-3 TrainSetBuild/build_trainset_pscpu.sbatch 100000 1000 42 4 /data/dust/user/zhaiyufe/TrainSet/ML_1D_Fitting_GISAXS_K1 1
#
# K=1 occupies 70%, K=2 occupies 30%:
# sbatch --array=0-3 TrainSetBuild/build_trainset_pscpu.sbatch \
#  100000 1000 45 4 \
#  /data/dust/user/zhaiyufe/TrainSet/ML_1D_Fitting_GISAXS_K1K2_p70p30 \
#  1,2 \
#  0.7,0.3
