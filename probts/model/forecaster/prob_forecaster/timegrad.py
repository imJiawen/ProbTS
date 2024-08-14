# ---------------------------------------------------------------------------------
# Portions of this file are derived from PyTorch-TS
# - Source: https://github.com/zalandoresearch/pytorch-ts
# - Paper: Multi-variate Probabilistic Time Series Forecasting via Conditioned Normalizing Flows
# - License: MIT, Apache-2.0 license

# We thank the authors for their contributions.
# ---------------------------------------------------------------------------------


import torch
import torch.nn as nn

from probts.data import ProbTSBatchData
from probts.utils import repeat
from probts.model.forecaster import Forecaster
from probts.model.nn import GaussianDiffusion
from probts.model.nn.layers.RevIN import RevIN

class TimeGrad(Forecaster):
    def __init__(
        self,
        enc_num_layers: int = 2,
        enc_hidden_size: int = 40,
        enc_dropout: float = 0.1,
        conditional_length: int = 100,
        beta_end: float = 0.1,
        diff_steps: int = 100,
        loss_type: str = "l2",
        beta_schedule: str = "linear",
        revin=False,
        affine=False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.autoregressive = True
        self.revin = revin
        if self.revin: self.revin_layer = RevIN(self.target_dim, affine=affine, subtract_last=False)
        
        self.encoder = nn.GRU(
            input_size=self.input_size,
            hidden_size=enc_hidden_size,
            num_layers=enc_num_layers,
            dropout=enc_dropout,
            batch_first=True
        )
        self.prob_model = GaussianDiffusion(
            target_dim=self.target_dim,
            f_hidden_size=enc_hidden_size,
            conditional_length=conditional_length,
            beta_end=beta_end,
            diff_steps=diff_steps,
            loss_type=loss_type,
            beta_schedule=beta_schedule
        )

    def loss(self, batch_data):
        if self.use_scaling:
            self.get_scale(batch_data)
            self.prob_model.scale = self.scaler.scale
        
        inputs = self.get_inputs(batch_data, 'all')
        if self.revin: 
            inputs[:,:,:self.target_dim] = self.revin_layer(inputs[:,:,:self.target_dim], 'norm')
        enc_outs, states = self.encoder(inputs)
        enc_outs = enc_outs[:, -self.prediction_length-1:-1, ...]
        
        dist_args = self.prob_model.dist_args(enc_outs)
        
        if self.revin: 
            target = self.revin_layer(batch_data.future_target_cdf, 'norm_only')
        else:
            target = batch_data.future_target_cdf
            
        loss = self.prob_model.loss(target, dist_args).unsqueeze(-1)
        loss = self.get_weighted_loss(batch_data, loss)
        return loss.mean()

    def forecast(self, batch_data, num_samples=None):
        if self.use_scaling:
            self.get_scale(batch_data)
        
        states = self.encode(batch_data)
        
        repeated_target_dimension_indicator = repeat(batch_data.target_dimension_indicator, num_samples)
        repeated_past_target_cdf = repeat(batch_data.past_target_cdf, num_samples)
        repeated_future_time_feat = repeat(batch_data.future_time_feat, num_samples)
        repeated_states = repeat(states, num_samples, dim=1)
        if self.use_scaling:
            repeated_scale = repeat(self.scaler.scale, num_samples)
            self.scaler.scale = repeated_scale
            self.prob_model.scale = repeated_scale

        future_samples = []
        for k in range(self.prediction_length):
            repeated_batch_data = ProbTSBatchData({
                'target_dimension_indicator': repeated_target_dimension_indicator,
                'past_target_cdf': repeated_past_target_cdf,
                'future_time_feat': repeated_future_time_feat[:, k:k+1, ...]
            }, device=batch_data.device)

            enc_outs, repeated_states = self.decode(repeated_batch_data, repeated_states, num_samples=num_samples)
            # Sample
            dist_args = self.prob_model.dist_args(enc_outs)
            new_samples = self.prob_model.sample(cond=dist_args)
            
            if self.revin: 
                new_samples = self.revin_layer(new_samples, 'denorm', num_samples=num_samples)
            future_samples.append(new_samples)

            repeated_past_target_cdf = torch.cat(
                (repeated_past_target_cdf, new_samples), dim=1
            )

        forecasts = torch.cat(future_samples, dim=1).reshape(
            -1, num_samples, self.prediction_length, self.target_dim)
        return forecasts

    def encode(self, batch_data):
        inputs = self.get_inputs(batch_data, 'encode')
        if self.revin: 
            inputs[:,:,:self.target_dim] = self.revin_layer(inputs[:,:,:self.target_dim], 'norm')
        outputs, states = self.encoder(inputs)
        return states

    def decode(self, batch_data, states=None, num_samples=None):
        inputs = self.get_inputs(batch_data, 'decode')
        if self.revin: 
            inputs[:,:,:self.target_dim] = self.revin_layer(inputs[:,:,:self.target_dim], 'norm_only',num_samples=num_samples)
        outputs, states = self.encoder(inputs, states)
        return outputs, states
