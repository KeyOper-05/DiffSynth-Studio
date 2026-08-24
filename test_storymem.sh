#!/usr/bin/env bash
set -e

export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export CPU_AFFINITY_CONF=1

# Each action uses the same cut=False training entry. Commands stay explicit so
# it is easy to remove or reorder actions without a Bash loop.

bash train_storymem_cut_false.sh flip > flip.log 2> flip_err.log

bash train_storymem_cut_false.sh 6am > 6am.log 2> 6am_err.log

bash train_storymem_cut_false.sh party > party.log 2> party_err.log
