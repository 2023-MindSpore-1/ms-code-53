#!/bin/bash

if [ $# -ne 2 ]
then
    echo "Usage: sh run_distribute_train_ascend.sh [DATA_PATH] [DEVICE_ID]"
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
export DEVICE_ID=$2
export RANK_ID=0
export RANK_SIZE=1

rm -rf ./train
mkdir ./train
cp ../*.py ./train
cp *.sh ./train
cp ../*.yaml ./train
cp -r ../src ./train
cd ./train || exit
echo "start training for device $DEVICE_ID"
env > env.log
python train.py --data_path=$PATH1 --output_path './output' > train.log 2>&1 &
cd ..
