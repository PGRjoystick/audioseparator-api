from fastapi import FastAPI, UploadFile, File
from audio_separator.separator import Separator
from starlette.responses import StreamingResponse
import os
import uvicorn
import torch
import subprocess
import uuid

app = FastAPI()

# Check if CUDA is available and print the result
cuda_available = torch.cuda.is_available()
print("CUDA is available: ", cuda_available)
# Print the PyTorch version
print("PyTorch version: ", torch.__version__)

@app.post("/separate_instrumental")
async def separate_instrumental(file: UploadFile = File(...)):
    # Save the uploaded file to disk
    filename = Path(file.filename).name
    unique_filename = str(uuid.uuid4()) + "_" + filename
    with open(unique_filename, "wb") as f:
        f.write(await file.read())


    # Initialize the Separator class
    separator = Separator(output_dir="output")

    # Load a machine learning model
    separator.load_model()

    # Perform the separation on the uploaded file
    output_files = separator.separate(unique_filename)

    # Find the instrumental file
    output_file_instrumental = next((f for f in output_files if "_(Instrumental)_" in f), None)
    if output_file_instrumental is None:
        raise Exception("Instrumental file not found")

    # Convert the instrumental output file to MP3
    file_path = os.path.join("output", output_file_instrumental)
    if os.path.exists(file_path):
        # Convert to MP3
        mp3_file_path = file_path.rsplit('.', 1)[0] + '.mp3'
        subprocess.run(['ffmpeg', '-i', file_path, mp3_file_path])
        os.remove(file_path)

    # Delete the original uploaded file
    os.remove(unique_filename)

    return StreamingResponse(open(mp3_file_path, "rb"), media_type="audio/mpeg")

@app.post("/separate_vocals")
async def separate_vocals(file: UploadFile = File(...)):
    # Save the uploaded file to disk
    filename = Path(file.filename).name
    unique_filename = str(uuid.uuid4()) + "_" + filename
    with open(unique_filename, "wb") as f:
        f.write(await file.read())

    # Initialize the Separator class
    separator = Separator(output_dir="output")

    # Load a machine learning model
    separator.load_model()

    # Perform the separation on the uploaded file
    output_files = separator.separate(unique_filename)

    # Find the vocal file
    output_file_vocals = next((f for f in output_files if "_(Vocals)_" in f), None)
    if output_file_vocals is None:
        raise Exception("Vocal file not found")

    # Convert the vocal output file to MP3
    file_path = os.path.join("output", output_file_vocals)
    if os.path.exists(file_path):
        # Convert to MP3
        mp3_file_path = file_path.rsplit('.', 1)[0] + '.mp3'
        subprocess.run(['ffmpeg', '-i', file_path, mp3_file_path])
        os.remove(file_path)

    # Delete the original uploaded file
    os.remove(unique_filename)

    return StreamingResponse(open(mp3_file_path, "rb"), media_type="audio/mpeg")

if __name__ == "__main__":
    uvicorn.run("seperate:app", host="0.0.0.0", port=8100, log_level="info")