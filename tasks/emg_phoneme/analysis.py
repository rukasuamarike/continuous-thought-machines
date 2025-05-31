"""
Analysis and testing tools for EMG phoneme recognition
"""

import sys
sys.path.append('..')

from models.ctm_emg import EMGContinuousThoughtMachine
from tasks.emg_phoneme.modules import OnsetDetector, apply_cwt_to_segments, prepare_features_for_ctm
from data.custom_datasets import EMGCTMDataset
from tasks.emg_phoneme.utils import preprocess_emg_data
from tasks.emg_phoneme.tester import run_comprehensive_pipeline_test
import json
# Import your comprehensive testing framework here
# (The EMGPipelineTester and related functions we created earlier)

def main():
    """Run comprehensive analysis of EMG-CTM pipeline"""
    
    # Configuration
    json_data_path = "/path/to/your/data"
    trial_names = ["trial_3", "trial_4"]
    

    # Run the comprehensive test
    phon = json.load(open("charsiu_phoneme.json"))
    phoneme_maps = {}
    for idx,key in enumerate(phon.keys()):
        phoneme_maps[key]=idx
        
    print("🔬 Running EMG-CTM Pipeline Analysis")
    print("=" * 50)
    
    # Run comprehensive testing
    success = run_comprehensive_pipeline_test(
        json_data_path, trial_names, phoneme_maps
    )

    
    if success:
        print("\n✅ All tests passed! Pipeline is ready for training.")
    else:
        print("\n❌ Some tests failed. Please check the issues above.")

if __name__ == "__main__":
    main()