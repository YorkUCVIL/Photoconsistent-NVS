#!/bin/bash
#SBATCH --nodes=2
#SBATCH --cpus-per-task=32
#SBATCH --mem=128gb
#SBATCH --gres=gpu:4
#SBATCH --partition=rtx6000
#SBATCH --qos=normal
#SBATCH --output=slurm-%j-%N.out

echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "
echo "Nodelist:= " $SLURM_JOB_NODELIST
echo "Number of nodes:= " $SLURM_JOB_NUM_NODES
echo "Ntasks per node:= "  $SLURM_NTASKS_PER_NODE
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "

source ~/.bashrc
conda activate photoconsistent-score

cd ../scripts

MAIN_HOST=`hostname -s`
export MASTER_ADDR=$MAIN_HOST
export MASTER_PORT=`python free-port.py $MASTER_ADDR`

export NCCL_IB_DISABLE=1

echo "MASTER_ADDR="$MASTER_ADDR
echo "MASTER_PORT="$MASTER_PORT

srun --nodes=$SLURM_JOB_NUM_NODES --ntasks-per-node=1 --mem=128gb python -u train.py --num_gpus_per_node 4 --num_nodes 2

wait
