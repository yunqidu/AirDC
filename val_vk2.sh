workspace="${AIRDC_WORKSPACE:-$(pwd)}"

log_dir="log/"
config_path="val_all_att_vk2.yaml"

GPU_num=$(nvidia-smi --list-gpus | wc -l)
MEM_MI=$(nvidia-smi --query-gpu=memory.total \
         --format=csv,noheader,nounits | head -n1)

VAL_PER_SAMPLE_MI=3500
val_batchsize=$(( MEM_MI / VAL_PER_SAMPLE_MI ))

if [ "$val_batchsize" -lt 4 ]; then
  val_batchsize=4
elif [ "$val_batchsize" -gt 12 ]; then
  val_batchsize=12
fi

NNODES=1
split="val"

echo "========= Current mode: $split ========="
echo "Nodes: $NNODES, GPUs per node: $GPU_num "
echo "val batchsize = $val_batchsize"

torchrun --nnodes=$NNODES --nproc_per_node=$GPU_num run.py \
        --workspace $workspace \
        --log_dir $log_dir \
        --config_path $config_path \
        --val_batchsize $val_batchsize \
        --override split=val \
        --no_debug \
        "$@"
