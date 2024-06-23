@echo off
python -m venv venv
call .\venv\Scripts\activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install onnxruntime
pip install onnxruntime-gpu
pip install "audio-separator[gpu]"
pip install fastapi
pip install python-dotenv
echo Setup complete.
python separate.py
pause
