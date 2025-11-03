#!/bin/bash

# Create directories if they do not exist
mkdir -p final_output
mkdir -p output
mkdir -p download
mkdir -p stereo_output
mkdir -p converted_output
mkdir -p boosted_converted_output
mkdir -p upload

# Check and setup .env file
if [ ! -f .env ]; then
    echo "============================================"
    echo "  Environment Configuration Setup"
    echo "============================================"
    echo ""
    echo "The .env file is missing. Let's set it up!"
    echo ""
    
    # Prompt for voice2voice endpoint
    echo "Enter your RVC/Voice2Voice server endpoint:"
    echo "(Example: http://localhost:8001/voice2voice)"
    echo "(Or press Enter for default: http://localhost:8001/voice2voice)"
    read -p "Endpoint: " user_endpoint
    
    # Use default if empty
    if [ -z "$user_endpoint" ]; then
        user_endpoint="http://localhost:8001/voice2voice"
    fi
    
    # Create .env file
    echo "VOICE2VOICE_ENDPOINT=$user_endpoint" > .env
    echo ""
    echo "✓ Created .env file with:"
    echo "  VOICE2VOICE_ENDPOINT=$user_endpoint"
    echo ""
else
    echo "✓ .env file exists"
    # Show current configuration
    if grep -q "VOICE2VOICE_ENDPOINT" .env; then
        current_endpoint=$(grep "VOICE2VOICE_ENDPOINT" .env | cut -d'=' -f2-)
        echo "  Current endpoint: $current_endpoint"
    fi
    echo ""
fi

# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
source ./venv/bin/activate

# Install dependencies in correct order to avoid conflicts
# Install compatible torch/torchaudio versions first (2.4.1 works with deepfilternet)
pip3 install "torch<2.5,>=2.4" "torchaudio<2.5,>=2.4" torchvision
pip install onnxruntime
pip install onnxruntime-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/
pip install "audio-separator[gpu]"
pip install fastapi python-dotenv uvicorn python-multipart deepfilternet

# Force numpy 2.x for audio-separator (deepfilternet works with it despite constraint)
pip install --upgrade --force-reinstall --no-deps "numpy>=2.0"

# Echo setup completion
echo "Setup complete."
echo ""

# Validate voice2voice endpoint connectivity (optional check)
if [ -f .env ]; then
    endpoint=$(grep "VOICE2VOICE_ENDPOINT" .env | cut -d'=' -f2-)
    if [ ! -z "$endpoint" ]; then
        # Extract host and port for connectivity check
        host=$(echo $endpoint | sed -E 's|https?://([^/:]+).*|\1|')
        port=$(echo $endpoint | sed -E 's|https?://[^:]+:([0-9]+).*|\1|')
        
        # If no port specified, use defaults
        if [ "$port" = "$endpoint" ]; then
            if [[ $endpoint == https://* ]]; then
                port=443
            else
                port=80
            fi
        fi
        
        echo "Checking connectivity to Voice2Voice server..."
        echo "  Host: $host"
        echo "  Port: $port"
        
        # Try to connect (timeout 3 seconds)
        if timeout 3 bash -c "echo > /dev/tcp/$host/$port" 2>/dev/null; then
            echo "✓ Voice2Voice server is reachable!"
        else
            echo "⚠ WARNING: Cannot connect to Voice2Voice server at $host:$port"
            echo "  Make sure your RVC/Voice2Voice server is running before using voice conversion features."
            echo "  You can update the endpoint in the .env file."
        fi
        echo ""
    fi
fi

# Run the separate.py script
echo "Starting Audio Separator API server..."
python separate.py