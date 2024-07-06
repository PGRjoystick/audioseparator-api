#!/bin/bash

# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Install dependencies
pip3 install torch torchvision torchaudio
pip install onnxruntime
pip install onnxruntime-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxrunti>pip install "audio-separator[gpu]"
pip install fastapi
pip install python-dotenv

# Echo setup completion
echo "Setup complete."

# Run the separate.py script
python separate.py