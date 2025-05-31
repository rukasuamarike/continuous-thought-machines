import torch
import torch.nn as nn
import numpy as np
from models.ctm import ContinuousThoughtMachine

from tasks.emg_phoneme.modules import EMGSelfAttentionBackbone

class ContinuousThoughtMachineEMG(ContinuousThoughtMachine):
    def set_backbone(self):
        """
        Set the backbone module based on the specified type.
        """
        if self.backbone_type == 'emg_self_attention':
            self.backbone = EMGSelfAttentionBackbone(
                n_channels=4,
                freq_bins=32, 
                time_steps=128,
                d_model=256,
                d_input=self.get_d_backbone(),
                n_heads=8,
                n_layers=2,
                dropout=0.1
            )
        else:
            raise ValueError(f"Invalid backbone_type: {self.backbone_type}")

    def get_d_backbone(self):
        """
        Get the dimensionality of the backbone output, to be used for positional embedding setup.

        This is a little bit complicated for resnets, but the logic should be easy enough to read below.        
        """
        if self.backbone_type == 'emg_self_attention':
            return 512
        else:
            raise ValueError(f"Invalid backbone_type: {self.backbone_type}")
    def set_initial_rgb(self):
        if self.backbone_type == 'emg_self_attention':
            self.initial_rgb = nn.Identity()  # EMG doesn't need RGB conversion
        else:
            self.initial_rgb = nn.Identity()
        return super().set_initial_rgb()
    
    def compute_features(self, x):
        if self.backbone_type == 'emg_self_attention':
        # For EMG: x is already CWT features (batch, 4, 32, 128)
        # Skip initial_rgb processing for EMG data
            self.kv_features = self.backbone(x)  # (batch, 512)
            
            # Handle positional embedding (likely 'none' for EMG)
            if self.positional_embedding_type != 'none':
                pos_emb = self.positional_embedding(self.kv_features)
                combined_features = self.kv_features + pos_emb
            else:
                combined_features = self.kv_features
                
            # Reshape for attention: EMG backbone outputs (batch, 512)
            # Need to add sequence dimension for attention
            kv = combined_features.unsqueeze(1)  # (batch, 1, 512)
            kv = self.kv_proj(kv)  # (batch, 1, d_input)
        else:
            raise ValueError(f"Invalid backbone_type: {self.backbone_type}")
        return kv