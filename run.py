
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import GradScaler
# sys.path.append(work_dir)
# sys.path.append(work_space+'model')
from utils import *
import copy

parse_config,config=parse_my_args()

# Local debug configuration.
if parse_config.debug:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use only '0' device
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"  # Since we're using a single process
    os.environ["LOCAL_RANK"] = "0"  # Local rank for the single GPU
    # Set the master address and port for distributed training
    os.environ["MASTER_ADDR"] = "localhost"  # Set the master address for local distributed execution.
    os.environ["MASTER_PORT"] = "13194"  # Can be any unused port
    # config.split='test_completion'
    config.split='val'
    config.split='train'
    config.train_batchsize=5
    config.mixed_precision=False
    # config.dataset='vkitti2'
    # config.train_strategy='iSLDV'
    # config.train_strategy='SLDV'
    config.depth_wise=False
    config.D=50
    config.vis_result=True
    config.lidar_lines=64
    config.confidence_guide=False
    config.update_with='selective'
    # config.update_with='igevplusplus'

config_val=copy.deepcopy(config)
config_val.crop_width  = config_val.owidth
config_val.crop_height = config_val.oheight
config_val.split='val'

split=config.split

# Initialize distributed training.
local_rank, rank, device = init_distributed()
print(f"local_rank: {local_rank}, rank: {rank}, device: {device}")

# Set random seeds after distributed initialization.
SEED = 2024  # Base seed.
set_seed(SEED, local_rank)  # Use a rank-specific seed for each process.
# Rank-specific seeds prevent identical data paths across processes.
# Avoid sampling collisions across batches.
# Rank-specific seeds improve effective parallel stochasticity.

#Set depth range parameters by dataset.
if config.dataset == 'kittidc':
    # Maximum depth.
    z_max = 100
    # Number of depth bins.
    D = config.D
elif config.dataset == 'vkitti2':
    z_max = 102
    D = 51
elif config.dataset == 'ms2':
    z_max = 100
    D = config.D

#Build model-specific arguments.
if config.model_name == "AirDC":
    excluded_keys = {"version","data_folder", "calib_folder", "baseline" , "ckpt_path"}  # Keys excluded from the config log.
    filtered_args_key = [k for k in config.keys() if k not in excluded_keys]
    filtered_args_value = [config[k] for k in filtered_args_key]

    args = config
    args.z_max=z_max
    args.D=D
    args_val=copy.deepcopy(args)
    args_val.crop_width = args_val.owidth
    args_val.crop_height = args_val.oheight
    import model.AirDC.airdc_model as mymodel
    model = mymodel.AirDC(args).to(device)

else:
    raise ValueError("model_name must be AirDC")

if config.split != "test_completion":
    model = setup_model(model, local_rank, find_unused_parameters=True)

#Log run metadata on the main process.
if rank== 0:
    print(f"Start with model")
    script_path = inspect.getfile(inspect.currentframe()) # Current script path.
    abs_script_path = os.path.abspath(script_path) # Absolute script path.
    print(f"Running script: {abs_script_path}")



if split == 'train':
    if config.dataset == 'kittidc':
        import utils.kitti_loader as loader
        train_dataset = loader.KittiDepth(args, 'train', return_left_right=True,)
    elif config.dataset == 'vkitti2':
        import utils.vkitti_loader as loader
        train_dataset = loader.VirtualKitti2(args, 'train')
    elif config.dataset == 'ms2':
        import utils.ms2_loader as loader
        train_dataset = loader.MS2Depth(args, 'train', return_left_right=True)
    if dist.is_initialized():
        train_sampler = DistributedSampler(train_dataset, shuffle=True)

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config.train_batchsize,
            num_workers=2,
            pin_memory=True,
            sampler=train_sampler,
            worker_init_fn=worker_init_fn,  # Initialize each DataLoader worker.
            collate_fn=custom_collate_fn_with_batch,
        )
    else:
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config.train_batchsize,
            num_workers=2,
            pin_memory=True,
            worker_init_fn=worker_init_fn,  # Initialize each DataLoader worker.
            collate_fn=custom_collate_fn_with_batch,
            shuffle=True
        )


    if rank== 0:
        print("\t==> train_loader size:", len(train_loader))
        
    if getattr(config, 'train_vis_ratio', None) is not None:
        train_num_samples_to_visualize = max(1, int(config.train_vis_ratio * len(train_loader)))
    elif config.dataset == 'kittidc':
        train_num_samples_to_visualize=4
    elif config.dataset == 'vkitti2':
        train_num_samples_to_visualize = max(1, int(0.05 * len(train_loader)))
    elif config.dataset == 'ms2':
        train_num_samples_to_visualize = 4
    else:
        train_num_samples_to_visualize = 1
        
    train_num_samples_to_visualize = min(len(train_loader), train_num_samples_to_visualize)
    train_visualize_indices = set(random.sample(range(len(train_loader)), train_num_samples_to_visualize))


if split in ['train','val','test_completion', 'test_prediction', 'test_compare']:
    # Import the KITTI Depth Completion dataset.
    if config.dataset == 'kittidc':
        import utils.kitti_loader as loader
        val_dataset = loader.KittiDepth(args, 'val', howtoval=config.howtoval,
                                        return_left_right=True)
    # Import the Virtual KITTI 2 dataset.
    elif config.dataset == 'vkitti2':
        import utils.vkitti_loader as loader
        val_dataset = loader.VirtualKitti2(args, 'val')
    elif config.dataset == 'ms2':
        import utils.ms2_loader as loader
        val_dataset = loader.MS2Depth(args_val, 'val', howtoval=config.howtoval, return_left_right=True)

    if dist.is_initialized():
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=config.val_batchsize,
            num_workers=4,
            pin_memory=True ,
            sampler=val_sampler,
            worker_init_fn=worker_init_fn,
            collate_fn=custom_collate_fn_with_batch,
        )
    else:
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=config.val_batchsize,
            num_workers=4,
            pin_memory=True,
            worker_init_fn=worker_init_fn,
            collate_fn=custom_collate_fn_with_batch,
            shuffle=False
        )
    if rank== 0:
        print("\t==> val_loader size:", len(val_loader))

    val_visualize_indices = set()
    if rank == 0:
        # Allow custom visualization ratio, fallback to save_result logic
        if getattr(config, 'vis_ratio', None) is not None:
            val_num_samples_to_visualize = max(1, int(config.vis_ratio * len(val_loader)))
        elif config.save_result:
            val_num_samples_to_visualize = max(1, int(1 * len(val_loader)))
        else:
            val_num_samples_to_visualize = max(1, int(0.1 * len(val_loader)))
    
        val_num_samples_to_visualize = min(len(val_loader), val_num_samples_to_visualize)
        val_visualize_indices = set(random.sample(range(len(val_loader)), val_num_samples_to_visualize))

if config.get("load_checkpoint_mode")==None or config.load_checkpoint_mode=="first":
    load_my_checkpoint_first(config,model,rank,debug=parse_config.debug)
elif config.load_checkpoint_mode=="merge":
    load_my_checkpoint_merge(config,model,rank,debug=parse_config.debug)
else:
    raise ValueError(f"❌ load_checkpoint_mode must be in [None,'first','merge'],but got {config.load_checkpoint_mode}")

#Back up the source code for training runs.
code_folder = config.workspace
src_folders = [
    Path(os.path.join(code_folder,"model")),
    Path(os.path.join(code_folder,"utils")),
    Path(os.path.join(code_folder,"func_tools")),
    Path(os.path.join(config.workspace, "config",config.config_path)),

]
log_folder=config.log_dir
#Save code backups only during training.
if rank== 0 and config.split=="train" : 
    backup_code(src_folders, log_folder)

train_csv_path = os.path.join(log_folder, "train_log.csv")
val_csv_path = os.path.join(log_folder, "val_log.csv")
test_csv_path=os.path.join(log_folder,"test_log.csv")
best_ckpt_path = os.path.join(log_folder, "best_model.pth")
latest_ckpt_path = os.path.join(log_folder, "latest_model.pth")

resume_epoch=0
if __name__ == "__main__":
    if split == 'train':
        iters_per_epoch = len(train_loader)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.lr,
            eps=1e-6,
            weight_decay=1e-4,  # Weight decay.
            betas=(0.9, 0.999)  # Default momentum parameters.
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, verbose=True)
        scaler = GradScaler()
        epochs = config.epochs
        criterion = DISPMSELoss_vis(args)
        for ep in range(resume_epoch+1, epochs):
            since = time.time()
            train_loss, val_loss = 0, 0
            train_metrics_accum = {"RMSE": 0, "MAE": 0, "iRMSE": 0, "iMAE": 0, 'near_RMSE': 0,
                         'near_MAE': 0, 'far_RMSE': 0, 'far_MAE': 0, 'near_iRMSE': 0, 'near_iMAE': 0,
                         'far_iRMSE': 0, 'far_iMAE': 0,"abs_rel": 0, "sq_rel": 0, "rms": 0, "log10": 0,
                         "delta1": 0, "delta2": 0, "delta3": 0}
            model.train()
            accumulation_steps = 1 # Number of mini-batches accumulated before an optimizer step.
            accumulated_loss = 0.0  # Initialize accumulated loss.

            for batch_idx, batch_data in enumerate(train_loader):
                print_on_master(ep, epochs, batch_idx, len(train_loader), since, get_lr(optimizer))
                batch_data = {k: v.to(local_rank) if not isinstance(v, list) else v for k, v in batch_data.items() }
                gt = batch_data["gt"].to(device)#(B,1,H,W)
                optimizer.zero_grad()
                torch.cuda.empty_cache()
                if hasattr(args, "update_with") and args.update_with == "igevplusplus":
                    (agg_predictions,depth_pred_iters,
                         disp_predictions,
                         masked_left,
                         masked_right,
                         sparse) = model(batch_data)
                    agg_predictions = [reshape_to_BHW(agg) for agg in agg_predictions]
                    depth_pred_iters = [reshape_to_BHW(depth) for depth in depth_pred_iters]
                else:
                    (depth_pred_iters,
                         disp_predictions,
                         masked_left,
                         masked_right,
                         sparse) = model(batch_data) #Return auxiliary tensors for loss visualization.
                    depth_pred_iters = [reshape_to_BHW(depth) for depth in depth_pred_iters]
                if disp_predictions is not None:
                    disp_predictions = [reshape_to_BHW(disp) for disp in disp_predictions]
                gt = reshape_to_BHW(gt)
                if any(check_nan_inf(output, f"output_{i}") for i, output in enumerate(depth_pred_iters)):
                    if rank== 0:
                        print("Problem detected in model outputs")
                    break
                loss = criterion(depth_pred_iters,gt,disp=disp_predictions,
                                 f=batch_data["P"][:,0,0],
                                 vis=batch_idx in train_visualize_indices,
                                 rgb_left=masked_left,
                                 rgb_right=masked_right,
                                 sparse=sparse,
                                 agg_preds=None if not hasattr(args, "update_with") or args.update_with != "igevplusplus" else agg_predictions
                                 )
                if check_nan_inf(loss, "loss"):
                    if rank== 0:
                        print("Problem detected in loss computation")
                    break

                # Periodically save a checkpoint during training.
                if rank== 0 and batch_idx in train_visualize_indices:
                    save_checkpoint(model, optimizer,scheduler, scaler, ep,
                                    batch_idx, train_loss,
                                    os.path.join(log_folder, "latest_model.pth"))
                                    
                # Save training visualizations locally.
                if config.vis_result and rank == 0 and batch_idx in train_visualize_indices:
                    epoch_str = f"_e{ep}"
                    visualize_all(args, depth_pred_iters[-1] if isinstance(depth_pred_iters, list) else depth_pred_iters,
                                  target=gt,
                                  save_path=os.path.join(log_folder, f"{config.dataset}_{config.model_name}_train_vis", f"vis_result_{batch_idx:03d}_{int(args.lidar_lines):02d}lines{epoch_str}.png"),
                                  rgb_left=batch_data.get("left_rgb"), rgb_right=batch_data.get("right_rgb"),
                                  sparse=batch_data.get("d"), split="train")

                accumulated_loss += loss
                if (batch_idx + 1) % accumulation_steps  == 0 :
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
                    scaler.scale(loss).backward()

                    try:
                        scaler.step(optimizer)
                    except RuntimeError as e:
                        print(f"Error: {e}")
                        raise

                    scaler.update()
                accumulated_loss = 0.0  # Reset accumulated loss.

                with (torch.cuda.amp.autocast(dtype=torch.float32, enabled=True)):
                    loss_value = torch.tensor(loss.item(), device=device)
                    torch.distributed.reduce(loss_value, dst=0, op=torch.distributed.ReduceOp.SUM)
                    #Aggregate loss.
                    if rank == 0:
                        loss_value /= dist.get_world_size()
                        train_loss += loss_value.item()
                    #Aggregate metrics.
                    train_batch_metrics=cal_metric_avg(depth_pred_iters,gt,rank,device)
                    if rank == 0:
                        for k, v in train_batch_metrics.items():
                            train_metrics_accum[k] += v
                    # Log training progress.
                    if rank== 0 and batch_idx > 0 and batch_idx % 20 == 0:
                        print(
                            f"Epoch [{ep}/{epochs}], Batch [{batch_idx}/{len(train_loader)}],"
                            f" Train Loss: {train_loss / (batch_idx + 1):.4f}"
                            f" Train rmse:{train_metrics_accum['RMSE'] / (batch_idx + 1):.4f}")
                    #Run validation at the end of the epoch.
                    if rank == 0 and  (batch_idx +1) == len(train_loader) :
                        #======================Save training metrics.======================
                        train_metrics_avg = {k: v / len(train_loader) for k, v in train_metrics_accum.items()}
                        if ep == resume_epoch+1:
                            csv_columns = ["Time", "Epoch", "train_batchsize", "TrainLoss", "LearningRate",
                                           "RMSE", "MAE", "iRMSE", "iMAE",
                                           "near_RMSE","near_MAE","near_iRMSE","near_iMAE",
                                        "far_RMSE","far_MAE","far_iRMSE","far_iMAE"] + filtered_args_key + ["TimeCost(min)"]  # Append config keys as CSV columns.
                            append_to_csv(train_csv_path, csv_columns)
                        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
                        train_row = [current_time,ep,config.train_batchsize, train_loss / len(train_loader),
                                    get_lr(optimizer)]
                        for metric in [ "RMSE", "MAE", "iRMSE", "iMAE",
                               "near_RMSE","near_MAE","near_iRMSE","near_iMAE",
                               "far_RMSE","far_MAE","far_iRMSE","far_iMAE"]:
                            train_row.append(train_metrics_avg[f"{metric}"])
                        train_row += filtered_args_value
                        train_row+=[(time.time() - since) / 60]
                        append_to_csv(train_csv_path, train_row)
                        
                        torch.cuda.empty_cache()
                        torch.save({
                            'epoch': ep,
                            'batch_idx': batch_idx,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict(),
                            'train_loss': train_loss / len(train_loader),
                        }, latest_ckpt_path)
            #======================Start validation.======================
            torch.distributed.barrier()  # Ensure all workers reach this point
            model1 = mymodel.AirDC(args_val).to(device)
            time.sleep(5)
            load_checkpoint_robust(model1,latest_ckpt_path,rank,debug_mode=parse_config.debug)
            current_val_loss = run_val(config_val,args_val, model1, val_loader,
                                       val_visualize_indices,device,
                                       single_val=False,criterion=criterion,
                                    optimizer=optimizer,epoch=ep,rank=rank,local_rank=local_rank)
            if rank== 0 :
                if current_val_loss < best_val_loss:
                    best_val_loss = current_val_loss
                    torch.save({
                        'epoch': ep,
                        'batch_idx': batch_idx,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'train_loss': train_loss / len(train_loader),
                        'loss': current_val_loss,
                    }, best_ckpt_path)
                print("Checkpoint saved.")
            torch.cuda.empty_cache()
            gc.collect()
    elif split == 'val':
        test_iterations = getattr(config, 'test_iterations', None)
        current_val_loss = run_val(config, args, model, val_loader,
                               val_visualize_indices,device,
                               single_val=True,rank=rank,local_rank=local_rank,
                               test_csv_path=test_csv_path,filtered_args_key=filtered_args_key,
                               filtered_args_value=filtered_args_value,
                               max_iterations=test_iterations)

    elif split in ['test_completion', 'test_compare']:
        model.eval()
        # model.enable_timing()
        total_time = 0.0
        total_mem = 0.0
        total_samples = 0
        warmup_iterations = getattr(config, 'warmup_iterations', 10)
        test_iterations = getattr(config, 'test_iterations', 200)
        batch_size = 1
        resolution = "1216x256"
        torch.backends.cudnn.benchmark = True  # Enable convolution algorithm benchmarking.
        torch.backends.cuda.matmul.allow_tf32 = True  # Enable TF32 acceleration.
        # Measure parameters and FLOPs.
        if rank== 0:
            try:
                from thop import profile
                # Get a representative input batch.
                sample_data = val_dataset[0]
                sample_data =  {k: v.unsqueeze(0).cuda() if isinstance(v, torch.Tensor) else v for k, v in sample_data.items()}

                # Compute parameter count and FLOPs.
                if config.model_name == 'AirDC':
                    macs, params = profile(model, inputs=(sample_data, "test_completion"), verbose=False)
                else:
                    macs, params = profile(model, inputs=(sample_data,), verbose=False)
                flops = macs * 2  # Convert MACs to FLOPs.
            except ImportError:
                print("thop not installed, using manual param count")
                params = sum(p.numel() for p in model.parameters())
                flops = 0
        else:
            params = 0
            flops = 0

                # Synchronize results in DDP mode.
        params = torch.tensor(params).to(local_rank)
        flops = torch.tensor(flops).to(local_rank)
        import torch.distributed as dist
        if dist.is_initialized():
            torch.distributed.broadcast(params, src=0)
            torch.distributed.broadcast(flops, src=0)

        # Convert units.
        params_m = params.item() / 1e6
        flops_g = flops.item() / 1e9
        # Initialize runtime statistics.
        time_stats = []
        mem_stats = []

        # Initialize CUDA event timers.
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)

        # Configure memory measurement.
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


        # Add a low-overhead progress bar.
        class CudaSafeProgress:
            def __init__(self, total, desc=""):
                self.total = total
                self.desc = desc
                self.current = 0
                self.start_time = time.time()

            def update(self):
                self.current += 1
                if rank== 0:  # Display only on the main process in distributed mode.
                    elapsed = time.time() - self.start_time
                    avg_time = elapsed / self.current
                    remaining = avg_time * (self.total - self.current)
                    progress = f"{self.desc} [{self.current}/{self.total}] {elapsed:.1f}s<{remaining:.1f}s"
                    sys.stdout.write(f"\r{progress.ljust(80)}")
                    sys.stdout.flush()

        # Initialize lightweight progress monitoring.
        progress = CudaSafeProgress(total=warmup_iterations+test_iterations, desc="Benchmark Progress")


        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16,
                                                      enabled=True if config.model_name=='AirDC' and args.mixed_precision else False):
            
            # Prepare a single-sample batch from the validation dataset.
            def get_batch_data(idx):
                b_data = val_dataset[idx]
                return {k: v.unsqueeze(0).cuda() if isinstance(v, torch.Tensor) else v for k, v in b_data.items()}

            # Warm-up phase.
            for idx_wr in range(warmup_iterations):
                batch_data = get_batch_data(idx_wr)
                
                if config.model_name == 'AirDC':
                    # with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
                    _ = model(batch_data, split="test_completion")
                else:
                    _ = model(batch_data)
                progress.update()
        if split == "test_compare":
            csv_file = "test_compare.csv"
            if rank == 0:
                with open(csv_file, 'w') as f:
                    f.write("Iteration,RMSE (mm),GPU Memory (GB),Inference Time (ms)\n")
            
            for i in range(7):
                if i == 0:
                    model.args.train_strategy = "SLDV"
                    model.args.num_iters = 1
                else:
                    model.args.train_strategy = "iSLDV"
                    model.args.num_iters = i
                
                if rank == 0:
                    print(f"\n=========================================================")
                    print(f">>> Processing Iteration {i}...")
                    print(f"=========================================================")
                
                # Run validation to obtain RMSE.
                rmse_val = "?"
                if rank == 0:
                    print(">>> Running Val Mode for RMSE...")

                # Suppress progress output during evaluation.
                current_val_loss = run_val(config_val, args_val, model, val_loader,
                                        val_visualize_indices, device,
                                        single_val=True, rank=rank, local_rank=local_rank,
                                        silent=True)

                # RMSE extraction placeholder for benchmark mode.
                rmse_val = "evaluating" # Placeholder until run_val returns RMSE explicitly.
                
                # Memory and latency benchmark.
                if rank == 0:
                    print(">>> Running Test Mode for GPU Memory & Inference Time...")
                time_count = 0
                total_mem = 0
                mem_stats = []
                time_stats = []
                
                for idx_wr in range(warmup_iterations):
                    batch_data = get_batch_data(idx_wr)
                    if config.model_name == 'AirDC':
                        _ = model(batch_data, split="test_completion")
                    else:
                        _ = model(batch_data)
                        
                for val_idx in range(test_iterations):
                    batch_data = get_batch_data(val_idx)
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                    
                    start = time.perf_counter()
                    if config.model_name == 'AirDC':
                        _ = model(batch_data, split="test_completion")
                    else:
                        _ = model(batch_data)
                    end = time.perf_counter()

                    time_count += end - start
                    peak_mem = torch.cuda.memory_allocated() / (1024 ** 2)
                    mem_stats.append(peak_mem)
                    time_stats.append(end - start)
                
                avg_time = time_count / test_iterations if test_iterations > 0 else 0
                max_mem = max(mem_stats) if mem_stats else 0
                
                mem_gb = max_mem / 1024
                
                if rank == 0:
                    print(f">>> Iteration {i} Completed!")
                    print(f"    RMSE: {rmse_val} mm")
                    print(f"    Mem:  {mem_gb:.2f} GB")
                    print(f"    Time: {avg_time * 1000:.2f} ms")
                    with open(csv_file, 'a') as f:
                        f.write(f"{i},{rmse_val},{mem_gb:.2f},{avg_time * 1000:.2f}\n")
                        
        else:
            # Main benchmark loop.
            time_count = 0

            for val_idx in range(test_iterations):
                # Prepare input data.
                batch_data = get_batch_data(val_idx)
                
                # Reset memory statistics.
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

                # Measure latency with CUDA events.
                start = time.perf_counter()

                with torch.no_grad():
                    if config.model_name == 'AirDC':
                        out = model(batch_data, split="test_completion")
                    else:
                        out = model(batch_data)
                end = time.perf_counter()

                time_count += end - start
                batch_size_num = batch_data["left_rgb"].size(0)
                
                del out, batch_data
                torch.cuda.empty_cache()
                
                if val_idx%20==0:
                    avg_time = time_count / (val_idx + 1)
                    avg_fps = 1 / avg_time
                    print(f'time: {avg_time}, fps: {avg_fps}')
                # Record peak memory usage (MB)
                peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)

                # Collect benchmark statistics.
                total_mem += peak_mem
                total_samples += batch_size_num
                time_stats.append(time_count)  # Collect latency statistics.
                mem_stats.append(peak_mem)
                progress.update()

        # Compute summary statistics.
        avg_time = time_count / test_iterations if test_iterations > 0 else 0
        avg_mem = total_mem / test_iterations if test_iterations > 0 else 0
        max_mem = max(mem_stats) if mem_stats else 0
        min_mem = min(mem_stats) if mem_stats else 0

        fps = 1 / avg_time if avg_time > 0 else 0
        throughput = total_samples / time_count if time_count > 0 else 0
        
        # Compute the standard deviation safely.
        std_time = np.std([t * 1000 for t in time_stats]) if time_stats else 0.0

        if rank== 0:
            sys.stdout.write("\r" + " " * 80 + "\r")
            sys.stdout.flush()
            print(f"\n **** Model Name :{config.model_name} ****")
            print(f"\n{' Benchmark Results ':=^80}")
            print(f"| {'Metric':<25} | {'Value':<48} |")
            print(f"| {'-' * 25} | {'-' * 48} |")
            print(f"| {'Parameters':<25} | {params_m:.2f} M |")
            print(f"| {'FLOPs':<25} | {flops_g:.2f} G |")
            print(f"| {'Input Resolution':<25} | {resolution:<48} |")
            print(f"| {'Batch Size':<25} | {batch_size:<48} |")
            print(
                f"| {'Avg Inference Time':<25} | {avg_time * 1000:.2f} ms ± {std_time:.2f} ms |")  # Report the measured latency standard deviation.
            print(f"| {'Peak GPU Memory':<25} | {max_mem:.2f} MB (Range: {min_mem:.2f}-{max_mem:.2f} MB) |")
            print(f"| {'FPS':<25} | {fps:.2f} frames/sec |")
            print(f"| {'Throughput':<25} | {throughput:.2f} images/sec |")
            print(f"{'':=^80}\n")
