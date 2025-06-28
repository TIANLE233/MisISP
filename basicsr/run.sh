#!/bin/bash
######################################################################
#
# A rudimentary Bash script.
# The `cpu mode` and `--auto_resume` are not supported.
#
# sh run.sh func.py    task.yml   expe.yml   debug      force_yml
#           [Required] [Required] [Required] [Optional] [Optional]
#
######################################################################

python_path="python3"
# devices=3 # 0,1,2,3
num_devices=1

CUDA_VISIBLE_DEVICES="$1" \
  nohup \
  "$python_path" \
  -u -m torch.distributed.run \
  --nproc_per_node="$num_devices" \
  --master_port="$2" \
  "$3" \
  -path_arch "$4" \
  -path_model "$5" \
  -path_data "$6" \
  -expe_opt "$7" \
  -task_opt "$8" \
  --launcher pytorch \
  >> "$9" 2>&1 &
