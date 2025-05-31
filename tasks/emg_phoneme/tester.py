import numpy as np
from tasks.emg_phoneme.modules import OnsetDetector, apply_cwt_to_segments,prepare_features_for_ctm
from tasks.emg_phoneme.utils import butter_bandpass, extract_phoneme_for_timewindow_onset_bias, remove_spike, apply_basic_cwt_to_segments, visualize_cwt_features
import torch
import json
import matplotlib.pyplot as plt
from pathlib import Path
import time
from collections import defaultdict
from tasks.emg_phoneme.backbone import EMGSelfAttentionBackbone
class EMGPipelineTester:
    """
    Comprehensive tester for the complete EMG processing pipeline
    """
    
    def __init__(self, json_data_path, trial_names, phoneme_maps, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.json_data_path = json_data_path
        self.trial_names = trial_names
        self.phoneme_maps = phoneme_maps
        self.device = device
        
        # Load your JSON dataset structure
        self.datasets = {}
        for trial_name in trial_names:
            with open(f"{json_data_path}/ds_{trial_name}.json", 'r') as f:
                trial_data = json.load(f)
                self.datasets[trial_name] = trial_data[trial_name]
        
        self.results = defaultdict(list)
        
    def test_dataset_loading(self):
        """Test 1: Validate dataset structure and file accessibility"""
        print("🧪 Test 1: Dataset Loading and Structure Validation")
        print("=" * 60)
        
        for trial_name, trial_data in self.datasets.items():
            print(f"\n📁 Testing trial: {trial_name}")
            
            # Validate required keys
            required_keys = ['path', 'prompts', 'sr', 'emg_files', 'text_files', 'wav_files']
            missing_keys = [key for key in required_keys if key not in trial_data]
            
            if missing_keys:
                print(f"❌ Missing keys: {missing_keys}")
                return False
            else:
                print(f"✅ All required keys present")
            
            # Validate file counts match
            n_emg = len(trial_data['emg_files'])
            n_text = len(trial_data['text_files'])
            n_wav = len(trial_data['wav_files'])
            
            print(f"📊 File counts - EMG: {n_emg}, TextGrid: {n_text}, WAV: {n_wav}")
            
            if not (n_emg == n_text == n_wav):
                print(f"❌ File count mismatch!")
                return False
            else:
                print(f"✅ File counts match")
            
            # Test file accessibility
            accessible_files = 0
            for i, (emg_file, text_file) in enumerate(zip(trial_data['emg_files'][:3], trial_data['text_files'][:3])):
                if Path(emg_file).exists() and Path(text_file).exists():
                    accessible_files += 1
                    print(f"✅ Files {i+1} accessible")
                else:
                    print(f"❌ Files {i+1} not accessible: {emg_file}, {text_file}")
            
            # Validate sampling rate
            sr = trial_data['sr']
            print(f"📈 Sampling rate: {sr} Hz")
            
            self.results['dataset_validation'].append({
                'trial': trial_name,
                'files_count': n_emg,
                'accessible_files': accessible_files,
                'sampling_rate': sr
            })
        
        print(f"\n✅ Dataset loading test completed successfully!")
        return True
    
    def test_single_file_processing(self, trial_name=None, file_index=0):
        """Test 2: Process a single EMG file through the complete pipeline"""
        print(f"\n🧪 Test 2: Single File Processing Pipeline")
        print("=" * 60)
        
        if trial_name is None:
            trial_name = list(self.datasets.keys())[0]
        
        trial_data = self.datasets[trial_name]
        emg_file = trial_data['emg_files'][file_index]
        textgrid_file = trial_data['text_files'][file_index]
        sr = trial_data['sr']
        
        print(f"📄 Processing: {Path(emg_file).name}")
        print(f"📄 TextGrid: {Path(textgrid_file).name}")
        
        try:
            # Step 1: Load and preprocess EMG data
            print(f"\n📊 Step 1: Loading EMG data...")
            emg_data = np.loadtxt(emg_file, delimiter=',')[:, 1:5]  # Your 4 channels
            print(f"✅ Raw EMG shape: {emg_data.shape}")
            
            # Step 2: Clean EMG data (your existing preprocessing)
            print(f"📊 Step 2: Cleaning EMG data...")
            for ch in range(4):
                chan = emg_data[:, ch]
                chan[0] = chan[1]
                emg_data[:, ch] = remove_spike(chan)
                emg_data[:, ch] = butter_bandpass(emg_data[:, ch], 0.5, 50, sr)
            print(f"✅ Cleaned EMG shape: {emg_data.shape}")
            
            # Step 3: Onset detection and segmentation
            print(f"📊 Step 3: Onset detection and segmentation...")
            detector = OnsetDetector(
                threshold_percent=15,
                min_distance_samples=int(0.2 * sr),
                window_size_samples=int(0.5 * sr)
            )
            
            onset_results = detector.detect_cross_channel_onsets(emg_data)
            segments = detector.segment_around_onsets(emg_data, onset_results)
            print(f"✅ Detected {onset_results['n_onsets']} onsets")
            print(f"✅ Created {len(segments)} segments")
            
            # Step 4: CWT processing
            print(f"📊 Step 4: CWT processing...")
            cwt_features = apply_basic_cwt_to_segments(segments, wavelet='gaus3', sr=sr)
            print(f"✅ Generated CWT features for {len(cwt_features)} segments")
            
            # Step 5: Prepare for CTM
            print(f"📊 Step 5: Preparing CTM-ready features...")
            ctm_ready_features = prepare_features_for_ctm(cwt_features)
            print(f"✅ Prepared {len(ctm_ready_features)} CTM-ready features")
            
            # Step 6: Extract phoneme labels
            print(f"📊 Step 6: Extracting phoneme labels...")
            segment_labels = []
            for segment_features in ctm_ready_features:
                segment_info = segment_features['original_segment']
                start_sample = segment_info['window_start']
                end_sample = segment_info['window_end']
                
                phoneme_label = extract_phoneme_for_timewindow_onset_bias(
                    textgrid_file, start_sample, end_sample, sr, self.phoneme_maps
                )
                segment_labels.append(phoneme_label)
            
            print(f"✅ Extracted {len(segment_labels)} phoneme labels")
            
            # Step 7: Test backbone
            print(f"📊 Step 7: Testing EMG backbone...")
            backbone = EMGSelfAttentionBackbone(
                n_channels=4, freq_bins=32, time_steps=128,
                d_model=256, d_input=512, n_heads=8, n_layers=2, dropout=0.1
            ).to(self.device)
            
            backbone_outputs = []
            for segment_features in ctm_ready_features:
                cwt_tensor = torch.FloatTensor(segment_features['features']).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    output = backbone(cwt_tensor)
                    backbone_outputs.append(output.cpu())
            
            print(f"✅ Processed {len(backbone_outputs)} segments through backbone")
            
            # Store results
            self.results['single_file_processing'].append({
                'trial': trial_name,
                'file_index': file_index,
                'emg_shape': emg_data.shape,
                'n_onsets': onset_results['n_onsets'],
                'n_segments': len(segments),
                'n_features': len(ctm_ready_features),
                'n_labels': len(segment_labels),
                'backbone_output_shape': backbone_outputs[0].shape if backbone_outputs else None,
                'phoneme_distribution': self._analyze_phoneme_distribution(segment_labels)
            })
            
            print(f"✅ Single file processing completed successfully!")
            return True, {
                'emg_data': emg_data,
                'segments': segments,
                'ctm_ready_features': ctm_ready_features,
                'segment_labels': segment_labels,
                'backbone_outputs': backbone_outputs
            }
            
        except Exception as e:
            print(f"❌ Error in single file processing: {str(e)}")
            import traceback
            traceback.print_exc()
            return False, None
    
    def test_batch_processing(self, trial_name=None, max_files=5):
        """Test 3: Process multiple files to test consistency and memory usage"""
        print(f"\n🧪 Test 3: Batch Processing Test")
        print("=" * 60)
        
        if trial_name is None:
            trial_name = list(self.datasets.keys())[0]
        
        trial_data = self.datasets[trial_name]
        n_files = min(max_files, len(trial_data['emg_files']))
        
        print(f"📊 Processing {n_files} files from {trial_name}")
        
        all_segments = []
        all_labels = []
        all_backbone_outputs = []
        processing_times = []
        
        # Initialize backbone once
        backbone = EMGSelfAttentionBackbone(
            n_channels=4, freq_bins=32, time_steps=128,
            d_model=256, d_input=512, n_heads=8, n_layers=2, dropout=0.1
        ).to(self.device)
        
        for i in range(n_files):
            print(f"\n📄 Processing file {i+1}/{n_files}")
            start_time = time.time()
            
            success, results = self.test_single_file_processing(trial_name, i)
            
            if success:
                all_segments.extend(results['ctm_ready_features'])
                all_labels.extend(results['segment_labels'])
                all_backbone_outputs.extend(results['backbone_outputs'])
                
                processing_time = time.time() - start_time
                processing_times.append(processing_time)
                print(f"✅ File {i+1} processed in {processing_time:.2f}s")
            else:
                print(f"❌ File {i+1} failed processing")
        
        # Analyze batch results
        print(f"\n📊 Batch Processing Summary:")
        print(f"✅ Total segments: {len(all_segments)}")
        print(f"✅ Total labels: {len(all_labels)}")
        print(f"✅ Total backbone outputs: {len(all_backbone_outputs)}")
        print(f"⏱️ Average processing time: {np.mean(processing_times):.2f}s")
        print(f"⏱️ Total processing time: {sum(processing_times):.2f}s")
        
        # Test batch processing through backbone
        print(f"\n📊 Testing batch backbone processing...")
        if len(all_segments) > 0:
            batch_size = min(8, len(all_segments))
            batch_features = []
            batch_labels = []
            
            for i in range(batch_size):
                features = torch.FloatTensor(all_segments[i]['features'])
                batch_features.append(features)
                batch_labels.append(all_labels[i])
            
            batch_tensor = torch.stack(batch_features).to(self.device)
            batch_labels_tensor = torch.LongTensor(batch_labels).to(self.device)
            
            with torch.no_grad():
                batch_output = backbone(batch_tensor)
            
            print(f"✅ Batch processing successful!")
            print(f"   Input shape: {batch_tensor.shape}")
            print(f"   Output shape: {batch_output.shape}")
            print(f"   Labels shape: {batch_labels_tensor.shape}")
            
            self.results['batch_processing'].append({
                'trial': trial_name,
                'n_files': n_files,
                'total_segments': len(all_segments),
                'avg_processing_time': np.mean(processing_times),
                'batch_input_shape': batch_tensor.shape,
                'batch_output_shape': batch_output.shape
            })
        
        return all_segments, all_labels, all_backbone_outputs
    
    def test_data_consistency(self, segments, labels):
        """Test 4: Validate data consistency and quality"""
        print(f"\n🧪 Test 4: Data Consistency and Quality Validation")
        print("=" * 60)
        
        # Test feature consistency
        feature_shapes = [seg['features'].shape for seg in segments]
        unique_shapes = set(feature_shapes)
        
        print(f"📊 Feature shape analysis:")
        print(f"   Unique shapes: {unique_shapes}")
        
        if len(unique_shapes) == 1:
            print(f"✅ All features have consistent shape: {list(unique_shapes)[0]}")
        else:
            print(f"❌ Inconsistent feature shapes found!")
            return False
        
        # Test label distribution
        label_counts = defaultdict(int)
        for label in labels:
            label_counts[label] += 1
        
        print(f"\n📊 Label distribution:")
        reverse_phoneme_map = {v: k for k, v in self.phoneme_maps.items()}
        
        sorted_labels = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)
        for label_id, count in sorted_labels[:10]:  # Top 10
            phoneme = reverse_phoneme_map.get(label_id, f'Unknown_{label_id}')
            print(f"   {phoneme}: {count} ({count/len(labels)*100:.1f}%)")
        
        # Check for data quality issues
        n_silence = label_counts.get(self.phoneme_maps.get('[SIL]', -1), 0)
        silence_ratio = n_silence / len(labels)
        
        print(f"\n🔍 Data quality checks:")
        print(f"   Silence ratio: {silence_ratio:.3f}")
        
        if silence_ratio > 0.7:
            print(f"⚠️ High silence ratio detected - might affect training")
        else:
            print(f"✅ Silence ratio within reasonable range")
        
        # Test feature value ranges
        sample_features = np.array([seg['features'] for seg in segments[:100]])
        
        print(f"\n📊 Feature statistics (sample of 100):")
        print(f"   Mean: {sample_features.mean():.4f}")
        print(f"   Std: {sample_features.std():.4f}")
        print(f"   Min: {sample_features.min():.4f}")
        print(f"   Max: {sample_features.max():.4f}")
        
        # Check for NaN or inf values
        has_nan = np.isnan(sample_features).any()
        has_inf = np.isinf(sample_features).any()
        
        if has_nan or has_inf:
            print(f"❌ Found NaN or Inf values in features!")
            return False
        else:
            print(f"✅ No NaN or Inf values detected")
        
        return True
    
    def test_memory_usage(self, segments, max_batch_size=32):
        """Test 5: Memory usage and performance analysis"""
        print(f"\n🧪 Test 5: Memory Usage and Performance Analysis")
        print("=" * 60)
        
        if not torch.cuda.is_available():
            print("⚠️ CUDA not available, skipping GPU memory test")
            return
        
        backbone = EMGSelfAttentionBackbone(
            n_channels=4, freq_bins=32, time_steps=128,
            d_model=256, d_input=512, n_heads=8, n_layers=2, dropout=0.1
        ).to(self.device)
        
        batch_sizes = [1, 4, 8, 16, min(max_batch_size, len(segments))]
        
        for batch_size in batch_sizes:
            if batch_size > len(segments):
                continue
                
            # Prepare batch
            batch_features = []
            for i in range(batch_size):
                features = torch.FloatTensor(segments[i]['features'])
                batch_features.append(features)
            
            batch_tensor = torch.stack(batch_features).to(self.device)
            
            # Measure memory before
            torch.cuda.empty_cache()
            mem_before = torch.cuda.memory_allocated()
            
            # Forward pass
            start_time = time.time()
            with torch.no_grad():
                output = backbone(batch_tensor)
            torch.cuda.synchronize()
            forward_time = time.time() - start_time
            
            # Measure memory after
            mem_after = torch.cuda.memory_allocated()
            mem_used = (mem_after - mem_before) / 1024 / 1024  # MB
            
            print(f"📊 Batch size {batch_size:2d}: "
                  f"Memory: {mem_used:6.1f}MB, "
                  f"Time: {forward_time*1000:6.1f}ms, "
                  f"Time/sample: {forward_time/batch_size*1000:5.1f}ms")
        
        print(f"✅ Memory usage test completed")
    
    def _analyze_phoneme_distribution(self, labels):
        """Helper function to analyze phoneme distribution"""
        label_counts = defaultdict(int)
        for label in labels:
            label_counts[label] += 1
        return dict(label_counts)
    
    def generate_summary_report(self):
        """Generate a comprehensive summary report"""
        print(f"\n📋 COMPREHENSIVE PIPELINE TEST SUMMARY")
        print("=" * 80)
        
        print(f"\n📊 Dataset Summary:")
        for result in self.results['dataset_validation']:
            print(f"   {result['trial']}: {result['files_count']} files, "
                  f"{result['accessible_files']} accessible, SR={result['sampling_rate']}Hz")
        
        if self.results['single_file_processing']:
            print(f"\n📊 Single File Processing:")
            for result in self.results['single_file_processing']:
                print(f"   {result['trial']} file {result['file_index']}: "
                      f"{result['n_segments']} segments, {result['n_features']} features")
        
        if self.results['batch_processing']:
            print(f"\n📊 Batch Processing:")
            for result in self.results['batch_processing']:
                print(f"   {result['trial']}: {result['n_files']} files, "
                      f"{result['total_segments']} total segments, "
                      f"{result['avg_processing_time']:.2f}s avg time")
        
        print(f"\n✅ All tests completed! Your pipeline is ready for CTM integration.")

# Usage example and comprehensive test
def run_comprehensive_pipeline_test(json_data_path, trial_names, phoneme_maps):
    """
    Run the complete pipeline test suite
    """
    print("🚀 Starting Comprehensive EMG Pipeline Test Suite")
    print("=" * 80)
    
    tester = EMGPipelineTester(json_data_path, trial_names, phoneme_maps)
    
    # Test 1: Dataset loading
    if not tester.test_dataset_loading():
        print("❌ Dataset loading failed! Stopping tests.")
        return False
    
    # Test 2: Single file processing
    success, single_results = tester.test_single_file_processing()
    if not success:
        print("❌ Single file processing failed! Stopping tests.")
        return False
    
    # Test 3: Batch processing
    all_segments, all_labels, all_outputs = tester.test_batch_processing(max_files=3)
    
    # Test 4: Data consistency
    if not tester.test_data_consistency(all_segments, all_labels):
        print("❌ Data consistency check failed!")
        return False
    
    # Test 5: Memory usage
    tester.test_memory_usage(all_segments)
    
    # Generate summary
    tester.generate_summary_report()
    
    return True

def test_onset_detection(emg_data, emg_sr=250):
    """
    Test function for the onset detection MVP
    """
    detector = OnsetDetector(
        threshold_percent=15,  # As mentioned in your sources
        min_distance_samples=int(0.2 * emg_sr),  # 200ms minimum distance
        window_size_samples=int(0.5 * emg_sr)    # 500ms windows
    )

    print(f"Testing onset detection on EMG data shape: {emg_data.shape}")
    print(f"Sampling rate: {emg_sr} Hz")

    # Detect onsets
    onset_results = detector.detect_cross_channel_onsets(emg_data)

    print(f"\nDetected {onset_results['n_onsets']} unified onsets")

    # Create segments
    segments = detector.segment_around_onsets(emg_data, onset_results)

    print(f"Created {len(segments)} segments for further processing")

    # Visualization
    fig, axes = plt.subplots(4, 1, figsize=(15, 10))
    fig.suptitle('EMG Onset Detection Results')

    time_axis = np.arange(emg_data.shape[0]) / emg_sr

    for ch in range(4):
        ax = axes[ch]

        # Plot original signal
        ax.plot(time_axis, emg_data[:, ch], 'b-', alpha=0.7, label='EMG Signal')

        # Plot envelope and threshold
        ch_result = onset_results['channel_results'][f'ch_{ch}']
        ax.plot(time_axis, ch_result['envelope'], 'r-', linewidth=2, label='Envelope')
        ax.axhline(y=ch_result['threshold'], color='g', linestyle='--', label='Threshold')

        # Mark channel-specific onsets
        if len(ch_result['onsets']) > 0:
            onset_times = ch_result['onsets'] / emg_sr
            ax.scatter(onset_times, ch_result['heights'],
                      color='red', s=50, zorder=5, label='Onsets')

        # Mark unified onsets that came from this channel
        for onset_info in onset_results['unified_onsets']:
            if onset_info['channel'] == ch:
                onset_time = onset_info['sample'] / emg_sr
                ax.axvline(x=onset_time, color='orange', linestyle='-',
                          linewidth=3, alpha=0.8, label='Unified Trigger')

        ax.set_ylabel(f'Channel {ch}')
        ax.legend()
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Time (s)')
    plt.tight_layout()
    plt.show()

    return onset_results, segments


def test_data_segment_transform(emg_data,emg_sr):
    """
    check a data sample segmented and transform by visualizations
    """
    onset_results, segments = test_onset_detection(emg_data, emg_sr)
    # Apply CWT to your segments
    print("Applying basic CWT to detected segments...")
    cwt_features = apply_basic_cwt_to_segments(segments, wavelet='gaus3', sr=emg_sr)

    print(f"\nGenerated CWT features for {len(cwt_features)} segments")

    # Visualize the first segment
    if len(cwt_features) > 0:
        visualize_cwt_features(cwt_features, segment_idx=0)