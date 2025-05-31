"""Example training script for EMG-CTM"""

import torch
from torch.utils.data import DataLoader
import json

from tasks.emg_phoneme.config import EMG_CTM_CONFIG, EMG_BACKBONE_CONFIG,TRAINING_CONFIG, DATASET_CONFIG
from data.custom_datasets import EMGCTMDataset
from models.ctm_emg import EMGContinuousThoughtMachine
from tasks.emg_phoneme.trainer import EMGCTMTrainer

def main():
    # Device setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load phoneme mapping
    phoneme_maps = {
        '[SIL]': 0, 'AA': 1, 'AE': 2, 'AH': 3, 'AO': 4, 'AW': 5,
        # ... add all your phoneme mappings
    }
    
    # Create datasets
    json_data_path = "/path/to/your/data"
    
    dataset = EMGCTMDataset(
        json_data_path=json_data_path,
        trial_names=DATASET_CONFIG['train_trials'],
        phoneme_maps=phoneme_maps,
        onset_config={},  # Use defaults
        cwt_config={},    # Use defaults  
        preprocess_config={}, # Use defaults
    )
    
    # Split into train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    # Data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=TRAINING_CONFIG['batch_size'],
        shuffle=True,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=TRAINING_CONFIG['batch_size'], 
        shuffle=False,
        num_workers=4
    )
    
    # Initialize model
    model = EMGContinuousThoughtMachine(**EMG_CTM_CONFIG)
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=TRAINING_CONFIG['learning_rate'],
        weight_decay=TRAINING_CONFIG['weight_decay']
    )
    
    total_steps = len(train_loader) * TRAINING_CONFIG['num_epochs']
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=TRAINING_CONFIG['learning_rate'],
        total_steps=total_steps,
        pct_start=TRAINING_CONFIG['warmup_steps'] / total_steps
    )
    
    # Initialize trainer
    trainer = EMGCTMTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        grad_clip_norm=TRAINING_CONFIG['grad_clip_norm']
    )
    
    # Train
    history = trainer.train(TRAINING_CONFIG['num_epochs'])
    
    print("Training completed!")
    print(f"Best validation accuracy: {max(history['val_acc']):.4f}")

if __name__ == "__main__":
    main()
