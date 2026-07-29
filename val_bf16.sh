workspace="${AIRDC_WORKSPACE:-$(pwd)}"

log_dir="log/"
config_path="val_all_att.yaml"

# 1. Detect the number of GPUs.
GPU_num=$(nvidia-smi --list-gpus | wc -l)
# 2. Read the total memory of the first GPU(MiB)
MEM_MI=$(nvidia-smi --query-gpu=memory.total \
         --format=csv,noheader,nounits | head -n1)
# 3. Estimate the maximum batch size with 5 GB per sample.
TRAIN_PER_SAMPLE_MI=3000


train_batchsize=$(( MEM_MI / TRAIN_PER_SAMPLE_MI ))

if [ "$train_batchsize" -lt 4 ]; then
  train_batchsize=4
elif [ "$train_batchsize" -gt 12 ]; then
  train_batchsize=12
fi
val_batchsize=$((train_batchsize + 4))

GPU_num=1


echo "========= Current mode: $split ========="
echo "Validation uses one node and one GPU."
echo "✅ train batchsize = $train_batchsize, val batchsize = $val_batchsize"


torchrun --nnodes=1 --nproc_per_node=$GPU_num run.py \
        --workspace $workspace\
        --log_dir $log_dir\
        --config_path $config_path\
        --no_debug\
