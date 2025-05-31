"""
Core modules for EMG phoneme recognition pipeline
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange, repeat
from scipy.signal import find_peaks, hilbert, butter, filtfilt
from scipy import signal
import pywt
from typing import Dict, List, Tuple, Any
# ======================== ONSET DETECTION ========================

class OnsetDetector:
    """
    Onset detection for EMG signals using energy-based method
    """
    
    def __init__(self, 
                 threshold_percent: float = 15,
                 min_distance_samples: int = 50,
                 window_size_samples: int = 125):
        self.threshold_percent = threshold_percent
        self.min_distance = min_distance_samples
        self.window_size = window_size_samples
        
    def detect_channel_onsets(self, channel_data: np.ndarray, sr: int = 250) -> Dict:
        """Detect onsets in a single EMG channel"""
        # Calculate signal energy using Hilbert transform
        analytic_signal = hilbert(channel_data)
        envelope = np.abs(analytic_signal)
        
        # Smooth the envelope
        smoothed_envelope = signal.savgol_filter(
            envelope, 
            window_length=min(51, len(envelope)//10), 
            polyorder=3
        )
        
        # Calculate threshold as percentage of peak energy
        peak_energy = np.max(smoothed_envelope)
        threshold = (self.threshold_percent / 100.0) * peak_energy
        
        # Find peaks above threshold
        onsets, properties = find_peaks(
            smoothed_envelope, 
            height=threshold,
            distance=self.min_distance
        )
        
        return {
            'onsets': onsets,
            'envelope': smoothed_envelope,
            'threshold': threshold,
            'peak_energy': peak_energy,
            'heights': properties['peak_heights'] if onsets.size > 0 else np.array([])
        }
    
    def detect_cross_channel_onsets(self, emg_data: np.ndarray) -> Dict:
        """Detect onsets across all channels and create unified triggers"""
        n_samples, n_channels = emg_data.shape
        
        # Detect onsets in each channel
        channel_results = {}
        all_onsets = []
        
        for ch in range(n_channels):
            result = self.detect_channel_onsets(emg_data[:, ch])
            channel_results[f'ch_{ch}'] = result
            
            # Add channel info to onsets
            for onset in result['onsets']:
                all_onsets.append({
                    'sample': onset,
                    'channel': ch,
                    'strength': result['envelope'][onset]
                })
        
        # Sort all onsets by time
        all_onsets.sort(key=lambda x: x['sample'])
        
        # Create unified trigger points
        unified_onsets = []
        if all_onsets:
            unified_onsets.append(all_onsets[0])
            
            for onset in all_onsets[1:]:
                if onset['sample'] - unified_onsets[-1]['sample'] >= self.min_distance:
                    unified_onsets.append(onset)
        
        return {
            'channel_results': channel_results,
            'unified_onsets': unified_onsets,
            'n_onsets': len(unified_onsets)
        }
    
    def segment_around_onsets(self, emg_data: np.ndarray, onset_results: Dict) -> List[Dict]:
        """Create time windows around detected onsets"""
        segments = []
        n_samples = emg_data.shape[0]
        
        for onset_info in onset_results['unified_onsets']:
            onset_sample = onset_info['sample']
            
            # Define window around onset
            start_idx = max(0, onset_sample - self.window_size // 2)
            end_idx = min(n_samples, onset_sample + self.window_size // 2)
            
            # Extract segment
            segment = emg_data[start_idx:end_idx, :]
            
            segments.append({
                'data': segment,
                'onset_sample': onset_sample,
                'window_start': start_idx,
                'window_end': end_idx,
                'trigger_channel': onset_info['channel'],
                'trigger_strength': onset_info['strength']
            })
        
        return segments

# ======================== CWT PROCESSING ========================

def apply_cwt_to_segments(segments: List[Dict], 
                         wavelet: str = 'gaus3', 
                         scales: np.ndarray = None, 
                         sr: int = 250) -> List[Dict]:
    """Apply CWT to segmented EMG data"""
    
    if scales is None:
        # Create scales for 1-50 Hz
        frequencies = np.logspace(np.log10(1), np.log10(50), 20)
        scales = pywt.frequency2scale(wavelet, frequencies/sr)
    
    cwt_features = []
    
    for i, segment in enumerate(segments):
        segment_data = segment['data']
        n_samples, n_channels = segment_data.shape
        
        # Apply CWT to each channel
        channel_cwts = []
        for ch in range(n_channels):
            coefficients, freqs = pywt.cwt(segment_data[:, ch], scales, wavelet, 1/sr)
            # Convert to power
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
    
    return cwt_features

def prepare_features_for_ctm(cwt_features: List[Dict], 
                           target_shape: Tuple[int, int] = (32, 128)) -> List[Dict]:
    """Prepare CWT features for CTM input"""
    from scipy.ndimage import zoom
    
    ctm_ready_features = []
    
    for i, cwt_feat in enumerate(cwt_features):
        cwt_data = cwt_feat['cwt_data']  # (scales, samples, channels)
        
        # Resize each channel to target shape
        channel_features = []
        for ch in range(cwt_data.shape[2]):
            channel_cwt = cwt_data[:, :, ch]  # (scales, samples)
            
            # Resize to target shape
            scale_factor_freq = target_shape[0] / channel_cwt.shape[0]
            scale_factor_time = target_shape[1] / channel_cwt.shape[1]
            
            resized = zoom(channel_cwt, 
                          (scale_factor_freq, scale_factor_time), 
                          mode='nearest')
            
            channel_features.append(resized)
        
        # Stack channels: Shape (channels, freq_bins, time_steps)
        feature_tensor = np.stack(channel_features, axis=0)
        
        ctm_ready_features.append({
            'features': feature_tensor,
            'original_segment': cwt_feat['segment_info'],
            'shape': feature_tensor.shape
        })
    
    return ctm_ready_features