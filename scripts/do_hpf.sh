#!/bin/sh

# Set PBS Directives
# Lines starting with "#PBS", before any shell commands are
# interpreted as command line arguments to qsub.
# Don't put any commands before the #PBS options or they will not work
#
#PBS -V # Export all environment variables from the qsub commands environment to the batch job.
#PBS -l mem=250gb # Amount of memory needed by each processor (ppn) in the job.
#PBS -d /lustre/aoc/projects/hera/mmolnar/robstat # Working directory (PBS_O_WORKDIR) set to your Lustre area
#PBS -m bea # Send email when Jobs end or abort
#PBS -l nodes=1:ppn=8 # default is 1 core on 1 node
#PBS -l walltime=48:00:00
#PBS -j oe
#PBS -o hpf.out

source ~/.bashrc

conda activate robstat

date

cd /lustre/aoc/projects/hera/mmolnar/robstat
python /lustre/aoc/projects/hera/mmolnar/robstat/scripts/hpf.py lstb_no_avg/idr2_lstb_14m_ee_1.40949.npz

date
