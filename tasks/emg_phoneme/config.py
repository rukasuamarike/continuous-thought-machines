# CTM Configuration for EMG Speech Recognition
EMG_CTM_CONFIG = {
    # Core CTM architecture
    'iterations': 75,           # T - internal ticks
    'd_model': 1024,           # D - CTM latent space 
    'd_input': 512,            # From EMG backbone output
    'heads': 8,                # Multi-head attention
    
    # Synchronization
    'n_synch_out': 512,        # D_out for output predictions
    'n_synch_action': 256,     # D_action for attention queries
    'neuron_select_type': 'random-pairing',
    'n_random_pairing_self': 32,
    
    # Architecture details
    'synapse_depth': 8,        # U-Net depth
    'memory_length': 30,       # M - history for NLMs (~120ms at 250Hz)
    'deep_nlms': True,
    'memory_hidden_dims': 64,
    'do_layernorm_nlm': False,
    
    # EMG-specific settings
    'backbone_type': 'emg_self_attention',
    'positional_embedding_type': 'none',  # Backbone handles spatial relationships
    
    # Output
    'out_dims': 38,            # Phoneme classes
    'prediction_reshaper': [-1],
    'dropout': 0.1,
}

# EMG Backbone Configuration
EMG_BACKBONE_CONFIG = {
    'n_channels': 4,           # EMG channels
    'freq_bins': 32,           # CWT frequency bins
    'time_steps': 128,         # CWT time steps
    'd_model': 256,            # Internal processing dimension
    'd_input': 512,            # Output for CTM
    'n_heads': 8,              # Multi-head attention
    'n_layers': 2,             # Self-attention layers
    'dropout': 0.1,
}

# Onset Detection Configuration
ONSET_DETECTOR_CONFIG = {
    'threshold_percent': 15,    # 15% of peak energy (from literature)
    'min_distance_samples': 50, # ~200ms at 250Hz
    'window_size_samples': 125, # ~500ms at 250Hz
}

# CWT Processing Configuration
CWT_CONFIG = {
    'wavelet': 'gaus3',        # MUAP-shaped wavelet
    'scales': None,            # Auto-generate for 1-50Hz
    'target_shape': (32, 128), # Frequency bins, time steps
}
TRAINING_CONFIG = {
    'batch_size': 32,          # Start smaller for stability
    'learning_rate': 1e-4,     # CTM paper recommendation
    'weight_decay': 1e-4,      # Light regularization
    'num_epochs': 100,
    'warmup_steps': 10000,     # CTM used 10k warmup
    'grad_clip_norm': 20.0,    # CTM paper specification
    'scheduler': 'cosine',     # Cosine annealing with warmup
}

DATASET_CONFIG = {
    'max_segments_per_trial': None,  # Limit to prevent overfitting
    'train_trials': ['trial_3', 'trial_4'],
    'val_split': 0.2,
    'sampling_rate': 250,      # Hz
}

PREPROCESSING_CONFIG = {
    'bandpass_low': 0.5,       # Hz
    'bandpass_high': 50,       # Hz
    'filter_order': 2,         # Butterworth filter order
    'spike_threshold': 600,    # For spike removal
    'spike_window': 2,         # Samples around spike
}