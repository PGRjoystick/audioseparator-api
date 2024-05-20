from fastapi import FastAPI, UploadFile, File
from audio_separator.separator import Separator
import os
import base64
import uvicorn
import torch

app = FastAPI()

# Initialize the Separator class
separator = Separator(output_dir="output")

# Load a machine learning model
separator.load_model()
# Check if CUDA is available and print the result
cuda_available = torch.cuda.is_available()
print("CUDA is available: ", cuda_available)
# Print the PyTorch version
print("PyTorch version: ", torch.__version__)

@app.post("/separate")
async def separate(file: UploadFile = File(...)):
    # Save the uploaded file to disk
    with open(file.filename, "wb") as f:
        f.write(await file.read())

    # Define output_dir before using it
    output_dir = "output"

    # Perform the separation on the uploaded file
    separator.output_single_stem = 'instrumental'
    output_files_instrumental = separator.separate(file.filename)

    # Convert the instrumental output files to base64
    output_files_instrumental_base64 = []
    for output_file in output_files_instrumental:
        file_path = os.path.join(output_dir, output_file)
        if os.path.exists(file_path):
            output_files_instrumental_base64.append(base64.b64encode(open(file_path, "rb").read()).decode())
            os.remove(file_path)

    # Perform the separation on the uploaded file for vocals
    separator.output_single_stem = 'vocals'
    output_files_vocals = separator.separate(file.filename)

    # Convert the vocal output files to base64
    output_files_vocals_base64 = []
    for output_file in output_files_vocals:
        file_path = os.path.join(output_dir, output_file)
        if os.path.exists(file_path):
            output_files_vocals_base64.append(base64.b64encode(open(file_path, "rb").read()).decode())
            os.remove(file_path)

    # Return the base64 encoded output files
    return {
            "output_files_instrumental": output_files_instrumental_base64,
            "output_files_vocals": output_files_vocals_base64
    }

if __name__ == "__main__":
    uvicorn.run("seperate:app", host="0.0.0.0", port=8100, log_level="info")