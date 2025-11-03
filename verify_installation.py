#!/usr/bin/env python3
"""
Installation verification script for Audio Separator API
Run this after installation to verify all dependencies are working correctly.
"""

import sys

def check_imports():
    """Check if all critical imports work"""
    errors = []
    warnings = []
    
    print("Checking dependencies...\n")
    
    # Check audio-separator
    try:
        from audio_separator.separator import Separator
        print("✓ audio-separator imported successfully")
    except Exception as e:
        errors.append(f"✗ audio-separator import failed: {e}")
    
    # Check deepfilternet
    try:
        from df import enhance, init_df
        print("✓ deepfilternet imported successfully")
    except Exception as e:
        errors.append(f"✗ deepfilternet import failed: {e}")
    
    # Check torch
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__} imported successfully")
        
        # Check CUDA
        if torch.cuda.is_available():
            print(f"✓ CUDA is available (GPU acceleration enabled)")
            print(f"  - CUDA version: {torch.version.cuda}")
            print(f"  - GPU: {torch.cuda.get_device_name(0)}")
        else:
            warnings.append("⚠ CUDA not available (will use CPU only)")
    except Exception as e:
        errors.append(f"✗ PyTorch import failed: {e}")
    
    # Check torchaudio
    try:
        import torchaudio
        print(f"✓ torchaudio {torchaudio.__version__} imported successfully")
    except Exception as e:
        errors.append(f"✗ torchaudio import failed: {e}")
    
    # Check numpy
    try:
        import numpy as np
        print(f"✓ numpy {np.__version__} imported successfully")
        
        # Verify numpy version for audio-separator
        major_version = int(np.__version__.split('.')[0])
        if major_version < 2:
            warnings.append(f"⚠ numpy version {np.__version__} < 2.0 (audio-separator requires >=2.0)")
    except Exception as e:
        errors.append(f"✗ numpy import failed: {e}")
    
    # Check FastAPI
    try:
        from fastapi import FastAPI
        print("✓ FastAPI imported successfully")
    except Exception as e:
        errors.append(f"✗ FastAPI import failed: {e}")
    
    # Check other dependencies
    try:
        import requests
        print("✓ requests imported successfully")
    except Exception as e:
        errors.append(f"✗ requests import failed: {e}")
    
    try:
        import yt_dlp
        print("✓ yt-dlp imported successfully")
    except Exception as e:
        errors.append(f"✗ yt-dlp import failed: {e}")
    
    # Print summary
    print("\n" + "="*50)
    print("VERIFICATION SUMMARY")
    print("="*50)
    
    if warnings:
        print(f"\n⚠ Warnings ({len(warnings)}):")
        for warning in warnings:
            print(f"  {warning}")
    
    if errors:
        print(f"\n✗ Errors ({len(errors)}):")
        for error in errors:
            print(f"  {error}")
        print("\n❌ Installation verification FAILED")
        print("Please check the errors above and reinstall dependencies.")
        return False
    else:
        if warnings:
            print("\n✅ Installation verification PASSED with warnings")
            print("The application should work, but check warnings above.")
        else:
            print("\n✅ Installation verification PASSED")
            print("All dependencies are correctly installed!")
        return True

if __name__ == "__main__":
    success = check_imports()
    sys.exit(0 if success else 1)
