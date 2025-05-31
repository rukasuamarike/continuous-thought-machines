"""EMG Self-Attention Backbone for CTM integration"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from typing import Tuple

class EMGSelfAttentionBackbone(nn.Module):
    """
    Self-attention backbone for EMG CWT features
    
    Processes multi-channel EMG CWT features using self-attention
    to capture cross-channel muscle coordination patterns.
    """
    
    def __init__(
        self, 
        n_channels: int = 4,
        freq_bins: int = 32, 
        time_steps: int = 128,
        d_model: int = 256,
        d_input: int = 512,
        n_heads: int = 8,
        n_layers: int = 2,
        dropout: float = 0.1,
        use_attention_pooling: bool = True
    ):
        super().__init__()
        
        self.n_channels = n_channels
        self.freq_bins = freq_bins
        self.time_steps = time_steps
        self.d_model = d_model
        self.d_input = d_input
        self.use_attention_pooling = use_attention_pooling
        
        # Project each channel's CWT features to d_model
        self.channel_projection = nn.Linear(freq_bins * time_steps, d_model)
        
        # Learnable channel embeddings
        self.channel_embeddings = nn.Parameter(torch.randn(n_channels, d_model))
        
        # Self-attention across channels
        self.attention_layers = nn.ModuleList([
            ChannelAttentionBlock(d_model, n_heads, dropout) 
            for _ in range(n_layers)
        ])
        
        # Final projection to CTM input dimension
        if use_attention_pooling:
            self.attention_pool = AttentionPooling(d_model, d_input)
        else:
            self.output_projection = nn.Sequential(
                nn.Linear(n_channels * d_model, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, d_input),
                nn.LayerNorm(d_input)
            )
    
    def forward(self, cwt_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            cwt_features: (batch, channels, freq_bins, time_steps)
        Returns:
            features: (batch, d_input) for CTM processing
        """
        batch_size = cwt_features.shape[0]
        
        # Flatten and project each channel
        channel_features = rearrange(cwt_features, 'b c f t -> b c (f t)')
        channel_embeddings = self.channel_projection(channel_features)
        
        # Add learnable channel embeddings
        channel_emb = repeat(self.channel_embeddings, 'c d -> b c d', b=batch_size)
        channel_embeddings = channel_embeddings + channel_emb
        
        # Self-attention across channels
        attended_features = channel_embeddings
        for attention_layer in self.attention_layers:
            attended_features = attention_layer(attended_features)
        
        # Pool to final representation
        if self.use_attention_pooling:
            features = self.attention_pool(attended_features)
        else:
            features = rearrange(attended_features, 'b c d -> b (c d)')
            features = self.output_projection(features)
        
        return features

class ChannelAttentionBlock(nn.Module):
    """Self-attention block for processing EMG channels"""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        
        self.self_attention = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention across channels
        attended, _ = self.self_attention(x, x, x)
        x = self.norm1(x + self.dropout(attended))
        
        # Feed-forward
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

class AttentionPooling(nn.Module):
    """Attention-based pooling to aggregate channel information"""
    
    def __init__(self, d_model: int, d_output: int):
        super().__init__()
        
        self.attention_weights = nn.Linear(d_model, 1)
        self.output_projection = nn.Linear(d_model, d_output)
        self.layer_norm = nn.LayerNorm(d_output)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute attention weights for each channel
        weights = self.attention_weights(x)  # (batch, channels, 1)
        weights = F.softmax(weights, dim=1)   # Normalize across channels
        
        # Weighted average of channels
        pooled = torch.sum(x * weights, dim=1)  # (batch, d_model)
        
        # Final projection
        output = self.layer_norm(self.output_projection(pooled))
        
        return output

class ConstrainedEMGBackbone(nn.Module):
    """
    EMG backbone with constrained receptive field following CTM principles
    """
    
    def __init__(self, 
                 n_channels=4,
                 freq_bins=32, 
                 time_steps=128,
                 d_input=512,
                 kernel_size=3,    # CONSTRAINED - small kernels
                 max_layers=3):    # CONSTRAINED - limited depth
        
        super().__init__()
        
        # Constrained convolutions (following CTM readme philosophy)
        self.constrained_conv = nn.Sequential(
            # Layer 1: Small receptive field
            nn.Conv2d(n_channels, 64, kernel_size=kernel_size, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Layer 2: Still constrained
            nn.Conv2d(64, 128, kernel_size=kernel_size, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # Layer 3: Final constrained layer
            nn.Conv2d(128, 256, kernel_size=kernel_size, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        
        # Calculate flattened size after convolutions
        conv_output_size = 256 * freq_bins * time_steps
        
        # Project to d_input with constraint
        self.constrained_projection = nn.Sequential(
            nn.Linear(conv_output_size, d_input * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_input * 2, d_input),
            nn.LayerNorm(d_input)
        )
    
    def forward(self, cwt_features):
        """
        Constrained forward pass to force reasoning
        
        Args:
            cwt_features: (batch, 4, 32, 128)
        Returns:
            features: (batch, 512)
        """
        # Apply constrained convolutions
        x = self.constrained_conv(cwt_features)
        
        # Flatten and project
        x = x.view(x.size(0), -1)
        features = self.constrained_projection(x)
        
        return features
