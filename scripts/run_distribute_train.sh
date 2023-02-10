#!/bin/bash

if [ $# -ne 2 ]
then
    echo "Usage: sh run_distribute_train_ascend.sh [RANK_TABLE_FILE] [DATA_PATH]"
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

if [ ! -f $PATH1 ]
then
    echo "error: RANK_TABLE_FILE=$PATH1 is not a file"
exit 1
fi

PATH2=$(get_real_path $2)
echo $PATH2
if [ ! -d $PATH2 ]
then
    echo "error: IMAGE_PATH=$PATH2 is not a file"
exit 1
fi

ulimit -u unlimited
export DEVICE_NUM=8
export RANK_SIZE=8
export RANK_TABLE_FILE=$PATH1

for((i=0; i<${DEVICE_NUM}; i++))
do
    export DEVICE_ID=$i
    export RANK_ID=$i
    rm -rf ./train_parallel$i
    mkdir ./train_parallel$i
    cp ../*.py ./train_parallel$i
    cp ../*.yaml ./train_parallel$i
    cp *.sh ./train_parallel$i
    cp -r ../src ./train_parallel$i
    cd ./train_parallel$i || exit
    echo "start training for rank $RANK_ID, device $DEVICE_ID"
    env > env.log

    python train.py \
    --run_distribute=True \
    --data_path=$PATH2 \
    --output_path './output' > log.txt 2>&1 &

    cd ../
done
