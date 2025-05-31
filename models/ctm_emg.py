"""
CTM integration for EMG phoneme recognition.
Extends the base CTM with EMG-specific backbone and processing.
"""

import torch
import torch.nn as nn
from models.ctm import ContinuousThoughtMachine
from tasks.emg_phoneme.backbone import EMGSelfAttentionBackbone

class EMGContinuousThoughtMachine(ContinuousThoughtMachine):
    """
    CTM extended for EMG phoneme recognition using self-attention backbone
    """
    
    def __init__(self, **kwargs):
        # Set EMG-specific defaults
        emg_defaults = {
            'backbone_type': 'emg_self_attention',
            'positional_embedding_type': 'none',
            'd_input': 512,
            'out_dims': 38,  # phoneme classes
        }
        
        # Update with any provided kwargs
        config = {**emg_defaults, **kwargs}
        super().__init__(**config)
    
    def get_d_backbone(self):
        """Get backbone output dimensionality"""
        if self.backbone_type == 'emg_self_attention':
            return 512
        return super().get_d_backbone()
    
    def set_backbone(self):
        """Set EMG self-attention backbone"""
        if self.backbone_type == 'emg_self_attention':
            self.backbone = EMGSelfAttentionBackbone(
                n_channels=4,
                freq_bins=32, 
                time_steps=128,
                d_model=256,
                d_input=512,
                n_heads=8,
                n_layers=2,
                dropout=0.1
            )
        else:
            super().set_backbone()
    
    def set_initial_rgb(self):
        """EMG doesn't need RGB conversion"""
        if self.backbone_type == 'emg_self_attention':
            self.initial_rgb = nn.Identity()
        else:
            super().set_initial_rgb()
    
    def compute_features(self, x):
        """
        Compute features for EMG data
        
        Args:
            x: EMG CWT features (batch, 4, 32, 128)
        """
        if self.backbone_type == 'emg_self_attention':
            # Skip initial_rgb for EMG data
            self.kv_features = self.backbone(x)  # (batch, 512)
            
            # Handle positional embedding
            if self.positional_embedding_type != 'none':
                pos_emb = self.positional_embedding(self.kv_features)
                combined_features = self.kv_features + pos_emb
            else:
                combined_features = self.kv_features
                
            # Add sequence dimension for attention
            kv = combined_features.unsqueeze(1)  # (batch, 1, 512)
            kv = self.kv_proj(kv)  # (batch, 1, d_input)
            
        else:
            # Original processing for other backbone types
            kv = super().compute_features(x)
            
        return kv

# Add EMG backbone to valid types
from models.constants import VALID_BACKBONE_TYPES
if 'emg_self_attention' not in VALID_BACKBONE_TYPES:
    VALID_BACKBONE_TYPES.append('emg_self_attention')
