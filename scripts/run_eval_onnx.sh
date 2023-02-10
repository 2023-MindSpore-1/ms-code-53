#!/bin/bash

echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash scripts/run_onnx_eval.sh DATA_PATH ONNX_MODEL_PATH DEVICE_TYPE"
echo "for example: bash scripts/run_onnx_eval.sh /path/to/dataset /path/to/unet3d.onnx GPU "
echo "=============================================================================================================="

DATA_PATH=$1
ONNX_MODEL_PATH=$2
DEVICE_TYPE=$3

get_real_path(){
    if [ "${1:0:1}" == "/" ]; then
        echo "$1"
    else
        echo "$(realpath -m $PWD/$1)"
    fi
}

config_path=$(get_real_path "./default_config.yaml")
echo "config path is : ${config_path}"

python eval_onnx.py \
    --config_path=$config_path \
    --data_path=$DATA_PATH \
    --device_target=$DEVICE_TYPE \
    --file_name=$ONNX_MODEL_PATH > output.eval_onnx.log 2>&1 &
