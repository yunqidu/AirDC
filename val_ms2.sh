workspace="${AIRDC_WORKSPACE:-$(pwd)}"

log_dir="log/"
config_path="val_all_att_ms2.yaml"

# 1. Detect the number of GPUs.
GPU_num=$(nvidia-smi --list-gpus | wc -l)
# 2. Read the total memory of the first GPU(MiB)
MEM_MI=$(nvidia-smi --query-gpu=memory.total \
         --format=csv,noheader,nounits | head -n1)
# 3. Estimate the maximum batch size with 5 GB per sample.
TRAIN_PER_SAMPLE_MI=5500

val_batchsize=$(( MEM_MI / TRAIN_PER_SAMPLE_MI + 2 ))

if [ "$val_batchsize" -lt 6 ]; then
  val_batchsize=6
elif [ "$val_batchsize" -gt 14 ]; then
  val_batchsize=14
fi

NNODES=1
split="val"

echo "========= Current mode: $split ========="
echo "✅ Nodes: $NNODES, GPUs per node: $GPU_num "
echo "✅ val batchsize = $val_batchsize"


torchrun --nnodes=$NNODES --nproc_per_node=$GPU_num run.py \
        --workspace $workspace \
        --log_dir $log_dir \
        --config_path $config_path \
        --val_batchsize $val_batchsize \
        --override split=val \
        --no_debug \
        "$@"
