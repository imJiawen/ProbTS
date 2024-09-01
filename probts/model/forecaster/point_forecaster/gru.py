import torch
import torch.nn as nn

from probts.data import ProbTSBatchData
from probts.utils import repeat
from probts.model.forecaster import Forecaster
from probts.model.nn.layers.RevIN import RevIN

class GRUForecaster(Forecaster):
    def __init__(
        self,
        num_layers: int = 2,
        f_hidden_size: int = 40,
        dropout: float = 0.1,
        revin: bool = False,
        affine: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.autoregressive = True
        
        self.model = nn.GRU(
            input_size=self.input_size,
            hidden_size=f_hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        self.linear = nn.Linear(f_hidden_size, self.target_dim)
        self.loss_fn = nn.MSELoss(reduction='none')
        self.scale = None
        
        self.revin = revin
        if self.revin: self.revin_layer = RevIN(self.target_dim, affine=affine, subtract_last=False)

    def loss(self, batch_data):
        if self.use_scaling:
            self.get_scale(batch_data)
            self.scale = self.scaler.scale
        inputs = self.get_inputs(batch_data, 'all')
        
        if self.revin: 
            inputs[:,:,:self.target_dim] = self.revin_layer(inputs[:,:,:self.target_dim], 'norm')
        
        outputs, _ = self.model(inputs)
        outputs = outputs[:, -self.prediction_length-1:-1, ...]
        outputs = self.linear(outputs)
        
        if self.revin: 
            target = self.revin_layer(batch_data.future_target_cdf, 'norm_only')
        else:
            target = batch_data.future_target_cdf
        
        if self.scale is not None:
            outputs *= self.scale
        
        loss = self.loss_fn(batch_data.future_target_cdf, outputs)
        loss = self.get_weighted_loss(batch_data, loss)
        return loss.mean()

    def forecast(self, batch_data, num_samples=None):
        forecasts = []
        if self.use_scaling:
            self.get_scale(batch_data)
            self.scale = self.scaler.scale
        states = self.encode(batch_data)
        past_target_cdf = batch_data.past_target_cdf
        
        for k in range(self.prediction_length):
            current_batch_data = ProbTSBatchData({
                'target_dimension_indicator': batch_data.target_dimension_indicator,
                'past_target_cdf': past_target_cdf,
                'future_time_feat': batch_data.future_time_feat[:, k : k + 1:, ...]
            }, device=batch_data.device)

            outputs, states = self.decode(current_batch_data, states)
            outputs = self.linear(outputs)
            
            if self.revin: 
                outputs = self.revin_layer(outputs, 'denorm')
            
            forecasts.append(outputs)

            past_target_cdf = torch.cat(
                (past_target_cdf, outputs), dim=1
            )

        forecasts = torch.cat(forecasts, dim=1).reshape(
            -1, self.prediction_length, self.target_dim)
        if self.scale is not None:
            forecasts *= self.scale
        return forecasts.unsqueeze(1)

    def encode(self, batch_data):
        inputs = self.get_inputs(batch_data, 'encode')
        if self.revin: 
            inputs[:,:,:self.target_dim] = self.revin_layer(inputs[:,:,:self.target_dim], 'norm')
        outputs, states = self.model(inputs)
        return states

    def decode(self, batch_data, states=None, num_samples=None):
        inputs = self.get_inputs(batch_data, 'decode')
        if self.revin: 
            inputs[:,:,:self.target_dim] = self.revin_layer(inputs[:,:,:self.target_dim], 'norm_only')
        outputs, states = self.model(inputs, states)
        return outputs, states
