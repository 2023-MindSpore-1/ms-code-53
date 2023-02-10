#!/bin/bash

if [ $# -ne 1 ]
then
    echo "Usage: sh run_distribute_train_gpu_fp32.sh [DATA_PATH]"
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
export DEVICE_NUM=1
export DEVICE_ID=0
export RANK_ID=0
export RANK_SIZE=1

if [ -d "train_fp32" ];
then
    rm -rf ./train_fp32
fi

rm -rf ./train_fp32
mkdir ./train_fp32
cp ../*.py ./train_fp32
cp *.sh ./train_fp32
cp ../*.yaml ./train_fp32
cp -r ../src ./train_fp32
cd ./train_fp32 || exit
echo "start training for device $DEVICE_ID"
env > env.log
python train.py --data_path=$PATH1 --output_path './output' --device_target='GPU' --checkpoint_path='./' > train.log 2>&1 &
cd ..