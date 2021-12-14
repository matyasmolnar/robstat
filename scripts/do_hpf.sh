#!/bin/sh

# Set SBATCH Directives
# Lines starting with "#SBATCH", before any shell commands are
# interpreted as command line arguments to sbatch.
# Don't put any commands before the #SBATCH directives or they will not work.
#
#SBATCH --export=ALL                                  # Export all environment variables to job
#SBATCH --partition=hera                              # Specify partition on which to run job
#SBATCH --mem=100G                                    # Amount of memory needed by the whole job
#SBATCH -D /lustre/aoc/projects/hera/mmolnar/robstat  # Working directory
#SBATCH --mail-type=BEGIN,END,FAIL                    # Send email on begin, end, and fail of job
#SBATCH --nodes=1                                     # Request 1 node
#SBATCH --ntasks-per-node=8                           # Request 8 cores
#SBATCH --time=48:00:00                               # Request 48 hours, 0 minutes and 0 seconds.
#SBATCH --output=hpf.out

source ~/.bashrc

conda activate robstat

date

cd /lustre/aoc/projects/hera/mmolnar/robstat
python /lustre/aoc/projects/hera/mmolnar/robstat/scripts/hpf.py lstb_no_avg/idr2_lstb_14m_ee_1.40949.npz

date
