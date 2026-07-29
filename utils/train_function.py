import time
import datetime
import torch
from .basic import reshape_to_BHW
from .fit_utils import print_on_master,append_to_csv
from .metrics import evaluate_metrics
from .visual_loss import visualize_all
import torch.distributed as dist
import os


def cal_metric_avg(depth_pred_iters, gt, rank, device):
    """
    Compute all metrics, including absolute/relative errors, log10, and delta accuracies, using distributed sum-and-average across GPUs.
    :param depth_pred_iters: model outputs as a list or Tensor with shape (B, H, W), in meters.
    :param gt: ground-truth depth Tensor with shape (B, H, W), in meters.
    :param rank: current process rank; rank 0 returns averaged results.
    :param device: target device, e.g., "cuda:0".
    :return: metric averages on rank 0; an empty dictionary on other ranks.
    """

    # Define the metric names used for evaluation.
    metric_names = [
        # Global and near/far metrics.
        "RMSE", "MAE", "iRMSE", "iMAE",
        "near_RMSE", "near_MAE", "near_iRMSE", "near_iMAE",
        "far_RMSE", "far_MAE", "far_iRMSE", "far_iMAE",
        # Common relative-depth metrics.
        "abs_rel", "sq_rel", "rms", "log10",
        "delta1", "delta2", "delta3"
    ]

    # Initialize all metric accumulators to zero.
    # These values are summed across processes and averaged on rank 0.
    return_metrics = {name: 0.0 for name in metric_names}

    # Evaluate the current batch.
    # evaluate_metrics returns a dictionary containing all metric_names.
    # The metric implementation covers abs_rel, sq_rel, rms, log10, and delta accuracies.
    if isinstance(depth_pred_iters, list):
        current_metrics = evaluate_metrics(depth_pred_iters[-1], gt)
    else:
        current_metrics = evaluate_metrics(depth_pred_iters, gt)

    # Reduce each scalar metric to rank 0.
    # Use ReduceOp.SUM and divide by world_size on rank 0.
    world_size = dist.get_world_size()
    for key in metric_names:
        # Read the scalar metric value.
        val = current_metrics[key]
        # Wrap the scalar as a tensor on the target device for communication.
        metric_tensor = torch.tensor(val, device=device, dtype=torch.float64)
        # All processes reduce their metric tensor to rank 0.
        dist.reduce(metric_tensor, dst=0, op=dist.ReduceOp.SUM)

        if rank == 0:
            # Rank 0 divides the summed value by world_size.
            metric_tensor /= world_size
            return_metrics[key] = metric_tensor.item()

    # Only rank 0 returns the complete metric dictionary.
    if rank == 0:
        return return_metrics
    else:
        return {}


def run_val(config,
        args,
        model,
        val_loader,
        val_visualize_indices,
        device,
        single_val=True,
        criterion=None,
        optimizer=None,
        epoch=-1,
        rank=0,
        local_rank=0,
        test_csv_path="",
        filtered_args_key=None,
        filtered_args_value=None,
        max_iterations=None):

    if single_val:
        current_ep=0
        epoch_sum=1
    else:
        current_ep=epoch
        epoch_sum=config.epochs
    since = time.time()
    # val_loss, val_acc, val_precision, val_recall, val_f1 = 0, 0, 0, 0, 0
    model.eval()
    val_metrics_accum = {"RMSE": 0, "MAE": 0, "iRMSE": 0, "iMAE": 0, 'near_RMSE': 0,
                         'near_MAE': 0, 'far_RMSE': 0, 'far_MAE': 0, 'near_iRMSE': 0, 'near_iMAE': 0,
                         'far_iRMSE': 0, 'far_iMAE': 0,"abs_rel": 0, "sq_rel": 0, "rms": 0, "log10": 0,
                         "delta1": 0, "delta2": 0, "delta3": 0}
    val_loss_accum = 0.0

    # with torch.cuda.amp.autocast(dtype=torch.float16, enabled=True if config.model_name=='AirDC' and args.mixed_precision else False):
    import torch
    with (torch.no_grad()):
        test_iterations = getattr(config, 'test_iterations', None)
        for index_v, batch_data in enumerate(val_loader):
            if max_iterations is not None and index_v >= max_iterations:
                break
            batch_data = {k: v.to(local_rank) if not isinstance(v, list) else v for k, v in batch_data.items()}
            print_on_master(current_ep, epoch_sum, index_v, len(val_loader), since,0.00)
            gt = batch_data["gt"].to(device)
            model = model.to(device)
            disp_predictions = None
            if config.model_name=='AirDC':
                if hasattr(args, "update_with") and args.update_with == "igevplusplus":
                    (agg_predictions, depth_pred_iters,
                     disp_predictions,
                     masked_left,
                     masked_right,
                     sparse) = model(batch_data)
                    agg_predictions = [reshape_to_BHW(agg) for agg in agg_predictions]
                else:
                    (depth_pred_iters,
                        disp_predictions,
                        masked_left,masked_right,sparse) = model(batch_data, split="val")
                depth_pred_iters = [reshape_to_BHW(depth) for depth in depth_pred_iters]
                if disp_predictions is not None:
                    disp_predictions = [reshape_to_BHW(disp) for disp in disp_predictions]
            else:
                raise ValueError("model_name must be AirDC")
            gt = reshape_to_BHW(gt)
            if disp_predictions is not None:
                disp_predictions = [reshape_to_BHW(disp) for disp in disp_predictions]
            # Compute validation loss only at the end of each epoch.
            if not single_val:
                loss = criterion(depth_pred_iters, gt,
                                 disp=disp_predictions, f=batch_data["P"][:, 0, 0],
                                 vis=index_v in val_visualize_indices,
                                 val=True,
                                 rgb_left=masked_left,
                                 rgb_right=masked_right,
                                 sparse=sparse,
                                 agg_preds=None if not hasattr(args, "update_with") or args.update_with != "igevplusplus" else agg_predictions
                                 )
                loss_value = torch.tensor(loss.item(), device=device)
                torch.distributed.reduce(loss_value, dst=0, op=torch.distributed.ReduceOp.SUM)
                # loss_value is the sum across ranks before division by world_size.
                if rank == 0:
                    loss_value /= dist.get_world_size()
                    val_loss_accum += loss_value.item()
            # Compute metrics on all GPUs.
            batch_metrics=cal_metric_avg(depth_pred_iters,gt,rank,device)
            if rank == 0:
                for k, v in batch_metrics.items():
                    val_metrics_accum[k] += v
            if rank == 0 and index_v % 20 == 0:
                print(
                    f"Epoch [{current_ep}/{epoch_sum}], Batch [{index_v}/{len(val_loader)}],"
                    f" Val Loss: {val_loss_accum / (index_v + 1):.4f}"
                    f" Val rmse:{val_metrics_accum['RMSE'] / (index_v + 1):.4f}")
            # Optionally save visualizations.
            if config.vis_result and index_v in val_visualize_indices:
                epoch_str = f"_e{current_ep}" if not single_val else "_single_val"
                visualize_all(args, depth_pred_iters[-1] if isinstance(depth_pred_iters,
                                                                        list) else depth_pred_iters,
                              target=gt,
                              save_path=os.path.join(config.log_dir, f"{config.dataset}_{config.model_name}_val_vis", f"vis_result_{index_v:03d}_{int(args.lidar_lines):02d}lines{epoch_str}.png"),
                              rgb_left=batch_data["left_rgb"], rgb_right=batch_data["right_rgb"],
                              sparse=batch_data["d"], split="val")
                              
        # Single validation run.
        if single_val and rank == 0:
            actual_iters = test_iterations if (test_iterations is not None and test_iterations < len(val_loader)) else len(val_loader)
            val_metrics_avg = {k: v / actual_iters for k, v in val_metrics_accum.items()}
            # Generate the current timestamp string.
            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
            # Add the timestamp as the first validation row field.
            csv_columns = ["Model Name","Time", "Epoch", "val_batchsize",
                            "RMSE", "MAE", "iRMSE", "iMAE",
                            "near_RMSE","near_MAE","near_iRMSE","near_iMAE",
                            "far_RMSE","far_MAE","far_iRMSE","far_iMAE",
                           "abs_rel","sq_rel","rms","log10","delta1","delta2","delta3"
                           ] + filtered_args_key + ["TimeCost(min)"]  # Append config keys as CSV columns.
            append_to_csv(test_csv_path, csv_columns)
            val_row = [config.model_name,current_time,"seperate_validation",config.val_batchsize, ]
            for metric in ["RMSE", "MAE", "iRMSE", "iMAE",
                            "near_RMSE","near_MAE","near_iRMSE","near_iMAE",
                            "far_RMSE","far_MAE","far_iRMSE","far_iMAE",
                           "abs_rel", "sq_rel", "rms", "log10", "delta1", "delta2", "delta3"
                           ]:
                val_row.append(val_metrics_avg[f"{metric}"])
            val_row += filtered_args_value
            val_row += [(time.time() - since) / 60]
            append_to_csv(test_csv_path, val_row)
            print(f" val rmse:{val_metrics_avg['RMSE']:.4f}")
        elif not single_val and rank == 0:
            actual_iters = test_iterations if (test_iterations is not None and test_iterations < len(val_loader)) else len(val_loader)
            val_metrics_avg = {k: v / actual_iters for k, v in val_metrics_accum.items()}
        actual_iters = test_iterations if (test_iterations is not None and test_iterations < len(val_loader)) else len(val_loader)
        return val_loss_accum / actual_iters #,val_metrics_avg["RMSE"]
