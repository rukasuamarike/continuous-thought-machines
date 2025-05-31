"""
Utilities for EMG phoneme recognition
"""

import numpy as np
import torch
from scipy.signal import butter, filtfilt
import json
from pathlib import Path
from typing import Dict, List, Tuple
from textgrid import TextGrid
import pywt
import matplotlib.pyplot as plt

def remove_spike(data: np.ndarray, w: int = 2, threshold: float = 600) -> np.ndarray:
    """Remove spikes from EMG data"""
    spike_indices = np.where(np.diff(data) > threshold)[0]
    for idx in spike_indices:
        if w < idx < len(data) - w:
            data[idx-w:idx+w] = np.repeat([(data[idx - w] + data[idx + w]) / 2], 2*w)
    return data

def butter_bandpass(data: np.ndarray, lowcut: float, highcut: float, 
                   fs: int, order: int = 2) -> np.ndarray:
    """Apply bandpass filter to EMG data"""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def preprocess_emg_data(emg_file: str, cfg:dict, sr: int = 250) -> np.ndarray:
    """Preprocess EMG data with filtering and artifact removal"""
    # Load EMG data
    data = np.loadtxt(emg_file, delimiter=',')
    emg_data = data[:, 1:5]  # Take 4 EMG channels
    
    # Clean each channel
    for ch in range(4):
        chan = emg_data[:, ch]
        chan[0] = chan[1]  # Fix first sample
        emg_data[:, ch] = remove_spike(chan,cfg['spike_window'],cfg['spike_threshold'])
        emg_data[:, ch] = butter_bandpass(emg_data[:, ch], cfg['bandpass_low'], cfg['bandpass_high'], sr,cfg['filter_order'])
    
    return emg_data

def extract_phoneme_for_timewindow_center(textgrid_file, start_sample, end_sample, emg_sr, phoneme_maps):
    """
    Alternative: Extract phoneme at the center of the time window
    """
    

    # Get center time of the window
    center_time = (start_sample + end_sample) / (2 * emg_sr)

    ts_data = TextGrid.fromFile(textgrid_file)

    # Find phoneme that contains the center point
    for interval in ts_data[0].intervals:
        if interval.minTime <= center_time <= interval.maxTime:
            return phoneme_maps.get(interval.mark, phoneme_maps.get("[SIL]", 0))

    # No phoneme found at center
    return phoneme_maps.get("[SIL]", 0)

def extract_phoneme_for_timewindow_onset_bias(textgrid_file, start_sample, end_sample, emg_sr, phoneme_maps):
    """
    Alternative: Bias towards phoneme that starts within the window (good for onset-triggered segments)
    """

    start_time = start_sample / emg_sr
    end_time = end_sample / emg_sr

    ts_data = TextGrid.fromFile(textgrid_file)

    # First, look for phonemes that start within our window
    for interval in ts_data[0].intervals:
        if start_time <= interval.minTime <= end_time:
            return phoneme_maps.get(interval.mark, phoneme_maps.get("[SIL]", 0))

    return extract_phoneme_for_timewindow_center(textgrid_file, start_sample, end_sample, emg_sr, phoneme_maps)



def apply_basic_cwt_to_segments(segments, scales=None, wavelet='gaus3', sr=250):
    """
    Apply basic CWT to segmented EMG data
    This is a simplified version before implementing fCWT

    Args:
        segments: List of segment dictionaries from onset detection
        scales: CWT scales to use (will auto-generate if None)
        wavelet: Wavelet to use ('gaus3' as you mentioned)
        sr: Sampling rate
    """
    if scales is None:
        # Create scales roughly corresponding to 1-50 Hz
        frequencies = np.logspace(np.log10(1), np.log10(50), 20)
        scales = pywt.frequency2scale(wavelet, frequencies/sr)

    cwt_features = []

    for i, segment in enumerate(segments):
        segment_data = segment['data']  # Shape: (samples, 4_channels)
        n_samples, n_channels = segment_data.shape

        # Apply CWT to each channel
        channel_cwts = []
        for ch in range(n_channels):
            # CWT for this channel
            coefficients, freqs = pywt.cwt(segment_data[:, ch], scales, wavelet, 1/sr)

            # Convert to power (magnitude squared)
            power = np.abs(coefficients)**2
            channel_cwts.append(power)

        # Stack all channels: Shape (scales, samples, channels)
        cwt_tensor = np.stack(channel_cwts, axis=-1)

        cwt_features.append({
            'cwt_data': cwt_tensor,
            'scales': scales,
            'frequencies': freqs,
            'segment_info': segment,
            'shape': cwt_tensor.shape
        })

        print(f"Segment {i}: CWT shape {cwt_tensor.shape} "
              f"(scales x samples x channels)")

    return cwt_features

def visualize_cwt_features(cwt_features, segment_idx=0):
    """
    Visualize CWT features for a specific segment
    """
    if segment_idx >= len(cwt_features):
        print(f"Segment {segment_idx} not available")
        return

    cwt_data = cwt_features[segment_idx]['cwt_data']
    freqs = cwt_features[segment_idx]['frequencies']
    segment_info = cwt_features[segment_idx]['segment_info']

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()

    for ch in range(4):
        ax = axes[ch]

        # Plot CWT coefficients for this channel
        im = ax.imshow(cwt_data[:, :, ch],
                      aspect='auto',
                      cmap='jet',
                      extent=[0, cwt_data.shape[1], freqs[-1], freqs[0]])

        ax.set_ylabel('Frequency (Hz)')
        ax.set_xlabel('Time (samples)')
        ax.set_title(f'Channel {ch} - CWT Power')
        plt.colorbar(im, ax=ax)

    plt.suptitle(f'CWT Features - Segment {segment_idx}\n'
                f'Triggered by Channel {segment_info["trigger_channel"]} '
                f'at sample {segment_info["onset_sample"]}')
    plt.tight_layout()
    plt.show()