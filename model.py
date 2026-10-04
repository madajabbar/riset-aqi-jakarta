"""
Implementation of N-BEATS (Neural Basis Expansion Analysis for Time Series).
Architecture: Oreshkin et al., 2020.
Used for modeling the residuals (non-linear components) after a linear baseline (SARIMA).
"""
import torch
import torch.nn as nn
import numpy as np


class NBeatsBlock(nn.Module):
    """A single building block of N-BEATS."""

    def __init__(self, input_size, hidden_size, theta_size, is_trend_block=True):
        super(NBeatsBlock, self).__init__()
        self.is_trend_block = is_trend_block

        # Hidden layer(s) -> Theta weights
        self.layers_stack = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, theta_size)
        )

        # Basis functions shape
        self.theta_size = theta_size
        self.num_steps_ahead = int(np.sqrt(theta_size)) * 2 if not is_trend_block else theta_size

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, input_size)
        Returns:
            forecast: Prediction output (scalar per batch item effectively)
        """
        # Get weights (theta) from history input
        theta = self.layers_stack(x)

        # Reconstruct forecast based on basis functions
        t = torch.linspace(0, self.num_steps_ahead - 1, self.num_steps_ahead).to(x.device)

        if self.is_trend_block:
            # Polynomial basis: t^0, t^1, ...
            basis_values = []
            k = min(self.num_steps_ahead, self.theta_size)
            for i in range(k):
                basis_values.append((t ** i) / (i + 1))  # Normalize power
            
            # If theta_size > num_basis_dims, pad or repeat (simplified logic)
            basis_matrix = torch.stack(basis_values[:self.theta_size], dim=-1) 
        else:
            # Trigonometric basis: sin, cos
            w = 2 * np.pi / self.num_steps_ahead
            sin_basis = torch.sin(w * t)
            cos_basis = torch.cos(w * t)
            basis_matrix = torch.cat([sin_basis, cos_basis], dim=-1)[:self.theta_size].unsqueeze(-1)

        # Forecast = dot product between learned weights and basis matrix
        # Note: In standard N-BEATS, basis depends on 'future steps' but here we simplify 
        # to predict 1-step ahead using static basis dimensions derived from theta_size.
        forecast = (theta.unsqueeze(-1) @ basis_matrix.unsqueeze(-1)).squeeze(-1)

        return forecast.squeeze()


class NBeatsModel(nn.Module):
    """
    N-BEATS Model consisting of multiple Stacks.
    Stack 1: Seasonality (Trigonometric)
    Stack 2: Trend (Polynomial)
    Residual connections between stacks.
    """

    def __init__(self, input_size, output_size, hidden_size=128, num_stacks=2,
                 num_layers_per_stack=2, sharesize=64):
        super(NBeatsModel, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.sharesize = sharesize

        # Stack configurations
        self.stack_configs = [
            {'trend': False},  # Seasonality stack first
            {'trend': True}    # Trend stack second
        ]

        self.stacks = nn.ModuleList()
        
        current_input = input_size

        for i, config in enumerate(self.stack_configs):
            is_trend = config['trend']
            
            blocks = nn.ModuleList()
            # Theta dimension usually equals input size
            theta_dim = input_size 

            for _ in range(num_layers_per_stack):
                blocks.append(NBeatsBlock(current_input, hidden_size, theta_dim, is_trend))
            
            self.stacks.append(blocks)
    
    def forward(self, x):
        """
        Args:
            x: Input history data (batch_size, input_size)
        Returns:
            forecast: Combined predictions (batch_size, 1)
        """
        residual = x
        forecasts = []
        
        for stack_idx, stack in enumerate(self.stacks):
            stack_forecasts = []
            for block in stack:
                # Block outputs forecast based on history
                pred = block(x) 
                stack_forecasts.append(pred)
            
            # Average aggregation across blocks within stack
            raw_preds = torch.stack(stack_forecasts, dim=1) 
            block_forecast = raw_preds.mean(dim=1) 
            
            forecasts.append(block_forecast)
            
            # Propagate residual to next stack
            if stack_idx < len(self.stacks) - 1:
                residual = residual - block_forecast
        
        # Sum all stack forecasts
        final_forecast = torch.sum(torch.stack(forecasts, dim=1), dim=1)
        return final_forecast