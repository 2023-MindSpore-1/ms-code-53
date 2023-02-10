#!/bin/bash

if [ $# -ne 1 ]
then
    echo "Usage: sh run_distribute_train_gpu.sh [DATA_PATH]"
exit 1
fi

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

PATH1=$(get_real_path $1)
echo $PATH1
if [ ! -d $PATH1 ]
then
    echo "error: IMAGE_PATH=$PATH1 is not a file"
exit 1
fi

ulimit -u unlimited
export DEVICE_NUM=8
export RANK_SIZE=8


if [ -d "train_parallel_fp32" ];
then
    rm -rf ./train_parallel_fp32
fi

rm -rf ./train_parallel_fp32
mkdir ./train_parallel_fp32
cp ../*.py ./train_parallel_fp32
cp *.sh ./train_parallel_fp32
cp ../*.yaml ./train_parallel_fp32
cp -r ../src ./train_parallel_fp32
cd ./train_parallel_fp32 || exit
echo "start distributed training with $DEVICE_NUM GPUs."
env > env.log
mpirun --allow-run-as-root -n $DEVICE_NUM --output-filename log_output --merge-stderr-to-stdout \
    python train.py --run_distribute=True --data_path=$PATH1 --output_path './output' --device_target='GPU' --checkpoint_path='./' > train.log 2>&1 &
cd ..
