@echo off
mkdir final_output
mkdir output
mkdir download
mkdir stereo_output
mkdir converted_output
mkdir boosted_converted_output
mkdir upload
python -m venv venv
call .\venv\Scripts\activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install onnxruntime
pip install onnxruntime-gpu
pip install "audio-separator[gpu]"
pip install fastapi python-dotenv uvicorn python-multipart
pip install deepfilternet requests

echo Setup complete.
python separate.py
pause
