# Setup Guide - Audio Separator API

## First Time Setup

When you run `./start.sh` for the first time, the script will automatically:

### 1. Create Required Directories
- `final_output/` - Final merged audio files
- `output/` - Temporary audio processing files
- `download/` - Downloaded YouTube content
- `stereo_output/` - Stereo converted audio
- `converted_output/` - Voice converted audio
- `boosted_converted_output/` - Volume boosted audio
- `upload/` - User uploaded files

### 2. Environment Configuration (Interactive)
If no `.env` file exists, you'll see:
```
============================================
  Environment Configuration Setup
============================================

The .env file is missing. Let's set it up!

Enter your RVC/Voice2Voice server endpoint:
(Example: http://localhost:8001/voice2voice)
(Or press Enter for default: http://localhost:8001/voice2voice)
Endpoint: 
```

**Options:**
- Press **Enter** for default: `http://localhost:8001/voice2voice`
- Enter **custom URL**: `http://192.168.1.100:8001/voice2voice`
- Enter **remote server**: `http://100.110.124.119:8001/voice2voice`

### 3. Python Virtual Environment
Creates and activates a virtual environment with all dependencies.

### 4. Dependency Installation
Installs packages in the correct order to avoid conflicts:
- Compatible PyTorch/torchaudio versions (2.4.1)
- ONNX Runtime for GPU acceleration
- Audio Separator with GPU support
- FastAPI web framework
- DeepFilterNet for noise reduction
- Correct numpy version (2.x) for compatibility

### 5. Server Connectivity Check
Before starting, the script checks if your Voice2Voice server is reachable:
```
Checking connectivity to Voice2Voice server...
  Host: 100.110.124.119
  Port: 8001
✓ Voice2Voice server is reachable!
```

Or warns you if not:
```
⚠ WARNING: Cannot connect to Voice2Voice server
  Make sure your RVC/Voice2Voice server is running
```

### 6. Start the Server
Launches the Audio Separator API on `http://0.0.0.0:8100`

---

## Reconfiguring Endpoint

If you need to change your Voice2Voice endpoint later:

### Method 1: Run the configuration script
```bash
./configure_endpoint.sh
```

### Method 2: Edit .env manually
```bash
nano .env
# Change the VOICE2VOICE_ENDPOINT value
# Save with Ctrl+X, Y, Enter
```

### Method 3: Delete .env and restart
```bash
rm .env
./start.sh
# You'll be prompted again
```

---

## Troubleshooting

### "Invalid URL 'None': No scheme supplied"
**Problem:** `.env` file is missing or `VOICE2VOICE_ENDPOINT` is not set  
**Solution:** Run `./configure_endpoint.sh` or `./start.sh`

### "Cannot connect to Voice2Voice server"
**Problem:** RVC server is not running or wrong endpoint  
**Solution:** 
1. Start your RVC server first
2. Verify the endpoint URL is correct
3. Check firewall/network connectivity
4. Run `./configure_endpoint.sh` to update endpoint

### Dependency conflicts during installation
**Problem:** Package version conflicts  
**Solution:** 
1. Delete the `venv/` directory: `rm -rf venv/`
2. Run `./start.sh` again (it will create a fresh environment)

### "ModuleNotFoundError: No module named 'torchaudio.backend'"
**Problem:** Wrong torchaudio version installed  
**Solution:** The updated `start.sh` fixes this automatically

---

## Verification

After setup completes, verify your installation:

```bash
# Activate the virtual environment
source venv/bin/activate

# Run the verification script
python verify_installation.py
```

This will check:
- ✓ All required packages are installed
- ✓ Correct versions are being used
- ✓ CUDA/GPU is available (if applicable)
- ✓ Dependencies are properly configured

---

## Common Endpoints Configuration

### Local Development
```bash
VOICE2VOICE_ENDPOINT=http://localhost:8001/voice2voice
```

### Local Network (same machine, different port)
```bash
VOICE2VOICE_ENDPOINT=http://127.0.0.1:8001/voice2voice
```

### Remote Server (LAN)
```bash
VOICE2VOICE_ENDPOINT=http://192.168.1.100:8001/voice2voice
```

### Remote Server (Tailscale/VPN)
```bash
VOICE2VOICE_ENDPOINT=http://100.110.124.119:8001/voice2voice
```

### Remote Server (Public IP)
```bash
VOICE2VOICE_ENDPOINT=http://203.0.113.42:8001/voice2voice
```

### HTTPS (if secured)
```bash
VOICE2VOICE_ENDPOINT=https://voice2voice.example.com/voice2voice
```

---

## Quick Start Commands

```bash
# First time setup
./start.sh

# Reconfigure endpoint
./configure_endpoint.sh

# Verify installation
source venv/bin/activate && python verify_installation.py

# Start server (after setup)
source venv/bin/activate && python separate.py

# Clean reinstall
rm -rf venv/ && ./start.sh
```

---

## Next Steps

After successful setup:
1. Your API server is running on `http://0.0.0.0:8100`
2. Visit `http://localhost:8100/docs` for API documentation
3. Make sure your RVC/Voice2Voice server is running
4. Test the endpoints using the examples in README.md
