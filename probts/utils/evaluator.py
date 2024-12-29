import numpy as np
from .metrics import *
import torch
from gluonts.ev.ts_stats import seasonal_error
from gluonts.time_feature.seasonality import get_seasonality
from einops import rearrange

class Evaluator:
    
    def __init__(self, quantiles_num=10, smooth=False):
        self.quantiles = (1.0 * np.arange(quantiles_num) / quantiles_num)[1:]
        self.ignore_invalid_values = True
        self.smooth = smooth

    def loss_name(self, q):
        return f"QuantileLoss[{q}]"

    def weighted_loss_name(self, q):
        return f"wQuantileLoss[{q}]"

    def coverage_name(self, q):
        return f"Coverage[{q}]"

    def get_sequence_metrics(self, targets, forecasts, seasonal_err=None, samples_dim=1,loss_weights=None):
        mean_forecasts = forecasts.mean(axis=samples_dim)
        median_forecasts = np.quantile(forecasts, 0.5, axis=samples_dim)

        metrics = {
            "MSE": mse(targets, mean_forecasts),
            "MAE": np.mean(abs_error(targets, mean_forecasts)),
            "sum_abs_error": np.sum(abs_error(targets, median_forecasts)),
            "abs_target_sum": abs_target_sum(targets),
            "abs_target_mean": abs_target_mean(targets),
            "MAPE": mape(targets, median_forecasts),
            "sMAPE": smape(targets, median_forecasts),
        }
        
        if seasonal_err is not None:
            metrics["MASE"] = mase(targets, median_forecasts, seasonal_err)
        
        metrics["RMSE"] = np.sqrt(metrics["MSE"])
        metrics["NRMSE"] = metrics["RMSE"] / metrics["abs_target_mean"] if metrics["abs_target_mean"] > 0 else 0
        metrics["ND"] = metrics["sum_abs_error"] / metrics["abs_target_sum"] if metrics["abs_target_sum"] > 0 else 0
        
        # calculate weighted loss
        if loss_weights is not None:
            nd = np.abs(targets - mean_forecasts) / np.sum(np.abs(targets), axis=(1, 2))
            loss_weights = loss_weights.detach().unsqueeze(0).unsqueeze(-1).numpy()
            weighted_ND = loss_weights * nd
            metrics['weighted_ND'] = np.sum(weighted_ND)
        else:
            metrics['weighted_ND'] = metrics["ND"]

        for q in self.quantiles:
            q_forecasts = np.quantile(forecasts, q, axis=samples_dim)
            metrics[self.loss_name(q)] = quantile_loss(targets, q_forecasts, q)
            metrics[self.weighted_loss_name(q)] = \
                metrics[self.loss_name(q)] / metrics["abs_target_sum"] if metrics["abs_target_sum"] > 0 else 0
            metrics[self.coverage_name(q)] = coverage(targets, q_forecasts)
        
        metrics["mean_absolute_QuantileLoss"] = np.mean(
            [metrics[self.loss_name(q)] for q in self.quantiles]
        )
        metrics["CRPS"] = np.mean(
            [metrics[self.weighted_loss_name(q)] for q in self.quantiles]
        )
        metrics["MAE_Coverage"] = np.mean(
            [
                np.abs(metrics[self.coverage_name(q)] - np.array([q]))
                for q in self.quantiles
            ]
        )
        return metrics

    def get_metrics(self, targets, forecasts, seasonal_err=None, samples_dim=1, loss_weights=None):
        metrics = {}
        seq_metrics = {}
        
        # Calculate metrics for each sequence
        for i in range(targets.shape[0]):
            single_seq_metrics = self.get_sequence_metrics(
                np.expand_dims(targets[i], axis=0),
                np.expand_dims(forecasts[i], axis=0),
                np.expand_dims(seasonal_err[i], axis=0) if seasonal_err is not None else None,
                samples_dim,
                loss_weights
            )
            for metric_name, metric_value in single_seq_metrics.items():
                if metric_name not in seq_metrics:
                    seq_metrics[metric_name] = []
                seq_metrics[metric_name].append(metric_value)
        
        for metric_name, metric_values in seq_metrics.items():
            metrics[metric_name] = np.mean(metric_values)
        return metrics

    @property
    def selected_metrics(self):
        return [ "ND", 'weighted_ND', 'CRPS', "NRMSE", "MSE", "MAE", "MASE", "MAPE", "sMAPE"]

    def __call__(self, targets, forecasts, past_data, freq, loss_weights=None, mode='multi'):
        """

        Parameters
        ----------
        targets
            groundtruth in (batch_size, prediction_length, target_dim)
        forecasts
            forecasts in (batch_size, num_samples, prediction_length, target_dim)
        Returns
        -------
        Dict[String, float]
            metrics
        """
        # if the input shape is [b, l]
        if len(targets.shape) == 2:
            mode='uni'
            
        targets = process_tensor(targets,mode=mode)
        forecasts = process_tensor(forecasts,mode=mode)
        past_data = process_tensor(past_data,mode=mode)
        
        if self.ignore_invalid_values:
            targets = np.ma.masked_invalid(targets)
            forecasts = np.ma.masked_invalid(forecasts)
        
        seasonality = get_seasonality(freq)
        
        # calculate seasonal error
        seasonal_err = seasonal_error(
            targets,
            seasonality=seasonality,
            time_axis=-2,
        )

        metrics = self.get_metrics(targets, forecasts, seasonal_err=seasonal_err, samples_dim=1, loss_weights=loss_weights)
        metrics_sum = self.get_metrics(targets.sum(axis=-1), forecasts.sum(axis=-1), samples_dim=1)
        
        # select output metrics
        output_metrics = dict()
        for k in self.selected_metrics:
            output_metrics[k] = metrics[k]
            if k in metrics_sum:
                output_metrics[f"{k}-Sum"] = metrics_sum[k]
                
        return output_metrics
    
def process_tensor(targets, mode='multi'):
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().detach().numpy()
    elif isinstance(targets, np.ndarray):
        pass 
    else:
        raise TypeError("targets must be a torch.Tensor or a numpy.ndarray")
    
    if mode == 'uni':
        if len(targets.shape) == 2:
            targets = rearrange(targets, 'b l -> b l 1')
        if len(targets.shape) == 3:
            targets = rearrange(targets, 'b l k -> (b k) l 1')
        if len(targets.shape) == 4:
            targets = rearrange(targets, 'b n l k -> (b k) n l 1')
    return targets