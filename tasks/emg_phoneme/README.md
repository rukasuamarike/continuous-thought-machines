# EMG Speech Recognition with Continuous Thought Machines

This package implements a biologically-inspired approach to EMG-based speech recognition using Continuous Thought Machines (CTMs). The system uses onset-triggered segmentation, continuous wavelet transforms, and neural synchronization for robust phoneme classification.

## Features

- **Onset-triggered segmentation**: Biologically-inspired signal segmentation based on neural activation patterns
- **CWT feature extraction**: Multi-resolution time-frequency analysis using wavelets shaped like Motor Unit Action Potentials (MUAPs)
- **Self-attention backbone**: Cross-channel muscle coordination modeling using attention mechanisms
- **CTM integration**: Neural synchronization as latent representation for temporal dynamics
- **Comprehensive testing**: Full pipeline validation and performance analysis tools

## Quick Start
```python 
from task.emg import EMGCTMDataset, EMGContinuousThoughtMachine, EMGCTMTrainer
# Load data
dataset = EMGCTMDataset( json_data_path="path/to/data", trial_names=["trial_3", "trial_4"], phoneme_maps=phoneme_mapping )
# Initialize model
model = EMGContinuousThoughtMachine()
# Train
trainer = EMGCTMTrainer(model, train_loader, val_loader, optimizer) history = trainer.train(num_epochs=100)
```


## Pipeline Overview

1. **Data Loading**: Load EMG data and TextGrid annotations
2. **Preprocessing**: Filter EMG signals and remove artifacts  
3. **Onset Detection**: Detect neural activation onsets across channels
4. **Segmentation**: Create time windows around detected onsets
5. **CWT Processing**: Apply continuous wavelet transform for time-frequency analysis
6. **Backbone Processing**: Use self-attention to model cross-channel coordination
7. **CTM Processing**: Leverage neural synchronization for phoneme classification

## Installation

```bash
# Clone the repository
git clone <repository-url> cd emg_speech
# Install dependencies
pip install -r requirements.txt
```



## Examples

See the `examples/` directory for:
- `train_emg_ctm.py`: Complete training pipeline
- `test_pipeline.py`: Pipeline validation and testing
- `demo_inference.py`: Real-time inference demonstration

