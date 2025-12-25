from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from audio_separator.separator import Separator
from starlette.responses import StreamingResponse
import os
import uvicorn
import torch
import subprocess
import uuid
from pathlib import Path
import requests
from typing import Optional
from fastapi.responses import JSONResponse
from fastapi.responses import StreamingResponse
from fastapi import UploadFile
from starlette.responses import StreamingResponse
from dotenv import load_dotenv
import requests
import re
from fastapi.responses import FileResponse
import shutil
import zipfile
from df import enhance, init_df
from df.io import load_audio, save_audio

app = FastAPI()
load_dotenv()
voice2voice_endpoint = os.getenv('VOICE2VOICE_ENDPOINT')

# Initialize DeepFilterNet model (lazy loading)
deepfilter_model = None
deepfilter_df_state = None
deepfilter_lock = None

def init_deepfilter():
    """Initialize DeepFilterNet model once (lazy loading)"""
    global deepfilter_model, deepfilter_df_state, deepfilter_lock
    if deepfilter_model is None:
        print("Initializing DeepFilterNet model...")
        from threading import Lock
        if deepfilter_lock is None:
            deepfilter_lock = Lock()
        with deepfilter_lock:
            if deepfilter_model is None:  # Double-check after acquiring lock
                deepfilter_model, deepfilter_df_state, _ = init_df()
                print("DeepFilterNet model initialized successfully")
    return deepfilter_model, deepfilter_df_state

def apply_deepfilter_noise_reduction(input_audio_path: str, output_audio_path: str) -> str:
    """
    Apply DeepFilterNet noise reduction to an audio file.
    
    Args:
        input_audio_path: Path to input audio file
        output_audio_path: Path to save denoised audio file
        
    Returns:
        Path to the denoised audio file
    """
    try:
        print(f"Applying DeepFilterNet noise reduction to: {input_audio_path}")
        
        # Initialize model (lazy loading)
        model, df_state = init_deepfilter()
        
        # Load audio
        audio, sr = load_audio(input_audio_path, sr=df_state.sr())
        
        # Enhance audio (remove noise)
        enhanced_audio = enhance(model, df_state, audio)
        
        # Save enhanced audio
        save_audio(output_audio_path, enhanced_audio, df_state.sr())
        
        print(f"DeepFilterNet noise reduction complete: {output_audio_path}")
        return output_audio_path
        
    except Exception as e:
        print(f"Error applying DeepFilterNet: {str(e)}")
        # If DeepFilterNet fails, return original file path
        print("Falling back to original audio without noise reduction")
        return input_audio_path

import time

def safe_file_remove(filename):
    for _ in range(5):  # try 5 times
        try:
            os.remove(filename)
            break
        except PermissionError:
            time.sleep(1)  # wait for 1 second before trying again

def file_streamer(filename):
    with open(filename, "rb") as f:
        yield from f
    safe_file_remove(filename)

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

    separator.output_single_stem = 'instrumental'

    # Load a machine learning model
    separator.load_model(model_filename='UVR-MDX-NET-Inst_full_292.onnx')

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

    separator.output_single_stem = 'vocals'

    # Load a machine learning model
    separator.load_model(model_filename='Kim_Vocal_2.onnx')

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

@app.post("/denoise")
async def denoise_audio(file: UploadFile = File(..., description="The audio file to denoise."),
                       output_format: str = Form("mp3", description="Output format: mp3, wav, or ogg")):
    """
    Apply DeepFilterNet3 noise reduction to an audio file without any other processing.
    
    This endpoint is useful for:
    - Removing background noise from recordings
    - Cleaning up audio before further processing
    - Enhancing speech clarity in noisy environments
    - Removing hiss, hum, or other unwanted sounds
    
    Returns the denoised audio in the specified format.
    """
    # Save the uploaded file to disk
    filename = Path(file.filename).name
    unique_filename = str(uuid.uuid4()) + "_" + filename
    upload_path = os.path.join("upload", unique_filename)
    
    # Ensure upload directory exists
    os.makedirs("upload", exist_ok=True)
    
    with open(upload_path, "wb") as f:
        f.write(await file.read())

    try:
        # Ensure output directory exists
        os.makedirs("final_output", exist_ok=True)
        
        # Apply DeepFilterNet noise reduction
        print(f"#################### Applying DeepFilterNet noise reduction to: {upload_path}")
        denoised_filename = f"denoised_{unique_filename.rsplit('.', 1)[0]}.wav"
        denoised_path = os.path.join("final_output", denoised_filename)
        
        apply_deepfilter_noise_reduction(upload_path, denoised_path)
        
        if not os.path.exists(denoised_path):
            raise HTTPException(status_code=500, detail="Noise reduction failed")
        
        print(f"#################### DeepFilterNet noise reduction complete: {denoised_path}")
        
        # Convert to requested format if not wav
        if output_format.lower() != "wav":
            output_filename = f"denoised_{unique_filename.rsplit('.', 1)[0]}.{output_format.lower()}"
            output_path = os.path.join("final_output", output_filename)
            
            # Set codec based on format
            if output_format.lower() == "mp3":
                codec_args = ['-c:a', 'libmp3lame', '-q:a', '2']
                media_type = "audio/mpeg"
            elif output_format.lower() == "ogg":
                codec_args = ['-c:a', 'libvorbis', '-q:a', '6']
                media_type = "audio/ogg"
            else:
                # Default to mp3 if format not recognized
                codec_args = ['-c:a', 'libmp3lame', '-q:a', '2']
                media_type = "audio/mpeg"
                output_format = "mp3"
            
            # Convert using FFmpeg
            subprocess.run(['ffmpeg', '-i', denoised_path, *codec_args, output_path], check=True)
            os.remove(denoised_path)  # Remove intermediate wav file
            final_path = output_path
        else:
            final_path = denoised_path
            media_type = "audio/wav"
        
        # Return the denoised audio file
        return StreamingResponse(
            open(final_path, "rb"), 
            media_type=media_type,
            headers={"Content-Disposition": f"attachment; filename=denoised_{filename.rsplit('.', 1)[0]}.{output_format.lower()}"}
        )
        
    except Exception as e:
        print(f"Error during noise reduction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Noise reduction failed: {str(e)}")
    
    finally:
        # Clean up the uploaded file
        if os.path.exists(upload_path):
            os.remove(upload_path)

@app.post("/denoise_youtube")
async def denoise_youtube(link: str = Form(..., description="YouTube video URL"),
                         output_format: str = Form("mp3", description="Output format: mp3, wav, or ogg")):
    """
    Download audio from a YouTube video and apply DeepFilterNet3 noise reduction.
    
    This is useful for:
    - Cleaning up audio from YouTube videos
    - Removing background noise from video content
    - Preparing YouTube audio for further processing
    
    Returns the denoised audio in the specified format.
    """
    try:
        # Download the YouTube audio
        print(f"Downloading audio from YouTube: {link}")
        filename = download_youtube_audio(link)
        
        if not os.path.exists(filename):
            raise HTTPException(status_code=500, detail="Failed to download YouTube audio")
        
        print(f"Downloaded audio file: {filename}")
        
        # Get the filename without the path for the UploadFile object
        audio_filename = os.path.basename(str(filename))
        
        # Call the denoise function with the downloaded file
        with open(filename, 'rb') as f:
            # Create an UploadFile object from the downloaded file with proper filename
            upload_file = UploadFile(filename=audio_filename, file=f)
            result = await denoise_audio(upload_file, output_format)
        
        # Clean up the downloaded file
        if os.path.exists(filename):
            os.remove(filename)
            print(f"Cleaned up downloaded file: {filename}")
        
        return result
        
    except Exception as e:
        print(f"Error during YouTube audio denoising: {str(e)}")
        # Clean up any downloaded files
        if 'filename' in locals() and os.path.exists(filename):
            os.remove(filename)
        raise HTTPException(status_code=500, detail=f"YouTube audio denoising failed: {str(e)}")

@app.post("/separate")
async def separate_6_stems(file: UploadFile = File(..., description="The audio file to separate into 6 stems.")):
    """
    Separate audio into 6 stems using htdemucs_6s model:
    - Vocals
    - Drums  
    - Bass
    - Guitar
    - Piano
    - Other
    
    Returns a ZIP file containing all 6 stems as OGG files (high quality).
    """
    # Save the uploaded file to disk
    filename = Path(file.filename).name
    unique_filename = str(uuid.uuid4()) + "_" + filename
    with open(unique_filename, "wb") as f:
        f.write(await file.read())

    try:
        # Initialize the Separator class
        separator = Separator(output_dir="output")
        
        # Load the htdemucs_6s model for 6-stem separation
        separator.load_model(model_filename='htdemucs_6s.yaml')
        
        # Perform the separation on the uploaded file
        output_files = separator.separate(unique_filename)
        print(f"6-stem separation complete! Output file(s): {' '.join(output_files)}")
        
        # Expected stem names from htdemucs_6s model
        stem_names = ['vocals', 'drums', 'bass', 'guitar', 'piano', 'other']
        stem_files = {}
        
        # Find all stem files
        for stem in stem_names:
            stem_file = next((f for f in output_files if f"_({stem.title()})_" in f), None)
            if stem_file:
                stem_files[stem] = stem_file
                print(f"Found {stem} stem: {stem_file}")
        
        if not stem_files:
            raise Exception("No stem files found after separation")
        
        # Convert all stems to OGG and prepare for ZIP
        ogg_files = []
        for stem, stem_file in stem_files.items():
            file_path = os.path.join("output", stem_file)
            if os.path.exists(file_path):
                # Convert to OGG with stem name (high quality compression)
                ogg_file_path = f"{stem}.ogg"
                subprocess.run(['ffmpeg', '-i', file_path, '-c:a', 'libvorbis', '-q:a', '6', ogg_file_path], check=True)
                ogg_files.append((ogg_file_path, stem))
                os.remove(file_path)  # Clean up original file
        
        # Ensure final_output directory exists
        final_output_dir = "final_output"
        os.makedirs(final_output_dir, exist_ok=True)
        
        # Create ZIP file containing all stems in final_output directory
        zip_filename = f"stems_{unique_filename.split('_')[0]}.zip"
        zip_filepath = os.path.join(final_output_dir, zip_filename)
        
        with zipfile.ZipFile(zip_filepath, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for ogg_file, stem_name in ogg_files:
                if os.path.exists(ogg_file):
                    zip_file.write(ogg_file, f"{stem_name}.ogg")
                    os.remove(ogg_file)  # Clean up individual OGG files
        
        print(f"ZIP file created: {zip_filepath}")
        
        # Return the ZIP file
        return StreamingResponse(
            open(zip_filepath, "rb"), 
            media_type="application/zip",
            headers={"Content-Disposition": f"attachment; filename={zip_filename}"}
        )
        
    except Exception as e:
        print(f"Error during 6-stem separation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Separation failed: {str(e)}")
    
    finally:
        # Clean up the uploaded file
        if os.path.exists(unique_filename):
            os.remove(unique_filename)
        
        # Clean up output directory
        if os.path.exists("output"):
            shutil.rmtree("output")
            os.makedirs("output", exist_ok=True)

@app.post("/remove_stems")
async def remove_stems(file: UploadFile = File(..., description="The audio file to process."),
                      vocals: bool = Form(False, description="Remove vocals from the audio"),
                      drums: bool = Form(False, description="Remove drums from the audio"),
                      bass: bool = Form(False, description="Remove bass from the audio"),
                      guitar: bool = Form(False, description="Remove guitar from the audio"),
                      piano: bool = Form(False, description="Remove piano from the audio"),
                      other: bool = Form(False, description="Remove other instruments from the audio")):
    """
    Remove selected stems from audio by separating into 6 stems and merging back without the selected ones.
    
    Uses htdemucs_6s model to separate into:
    - Vocals
    - Drums  
    - Bass
    - Guitar
    - Piano
    - Other
    
    Returns OGG audio file with the selected stems removed.
    """
    # Save the uploaded file to disk
    filename = Path(file.filename).name
    unique_filename = str(uuid.uuid4()) + "_" + filename
    with open(unique_filename, "wb") as f:
        f.write(await file.read())

    try:
        # Initialize the Separator class
        separator = Separator(output_dir="output")
        
        # Load the htdemucs_6s model for 6-stem separation
        separator.load_model(model_filename='htdemucs_6s.yaml')
        
        # Perform the separation on the uploaded file
        output_files = separator.separate(unique_filename)
        print(f"6-stem separation complete! Output file(s): {' '.join(output_files)}")
        
        # Expected stem names from htdemucs_6s model
        stem_names = ['vocals', 'drums', 'bass', 'guitar', 'piano', 'other']
        stem_files = {}
        
        # Find all stem files
        for stem in stem_names:
            stem_file = next((f for f in output_files if f"_({stem.title()})_" in f), None)
            if stem_file:
                stem_files[stem] = stem_file
                print(f"Found {stem} stem: {stem_file}")
        
        if not stem_files:
            raise Exception("No stem files found after separation")
        
        # Create mapping of parameters to stem removal
        stems_to_remove = {
            'vocals': vocals,
            'drums': drums,
            'bass': bass,
            'guitar': guitar,
            'piano': piano,
            'other': other
        }
        
        # Filter out stems that should be removed
        stems_to_keep = []
        for stem, stem_file in stem_files.items():
            if not stems_to_remove.get(stem, False):  # Keep if not marked for removal
                file_path = os.path.join("output", stem_file)
                if os.path.exists(file_path):
                    stems_to_keep.append(file_path)
                    print(f"Keeping {stem} stem")
            else:
                print(f"Removing {stem} stem")
        
        if not stems_to_keep:
            raise Exception("Cannot remove all stems - at least one stem must remain")
        
        # Ensure final_output directory exists
        final_output_dir = "final_output"
        os.makedirs(final_output_dir, exist_ok=True)
        
        # Create output filename
        output_filename = f"removed_stems_{unique_filename.split('_')[0]}.ogg"
        output_filepath = os.path.join(final_output_dir, output_filename)
        
        if len(stems_to_keep) == 1:
            # If only one stem remains, just convert it to OGG
            subprocess.run(['ffmpeg', '-i', stems_to_keep[0], '-c:a', 'libvorbis', '-q:a', '6', output_filepath], check=True)
        else:
            # Merge multiple stems using FFmpeg
            input_args = []
            for stem_path in stems_to_keep:
                input_args.extend(['-i', stem_path])
            
            # Build filter string for mixing
            inputs_count = len(stems_to_keep)
            filter_inputs = ''.join(f'[{i}:a]' for i in range(inputs_count))
            filter_complex = f'{filter_inputs}amix=inputs={inputs_count}:duration=first:dropout_transition=2[out]'
            
            command = [
                'ffmpeg',
                *input_args,
                '-filter_complex', filter_complex,
                '-map', '[out]',
                '-c:a', 'libvorbis',
                '-q:a', '6',
                output_filepath
            ]
            
            subprocess.run(command, check=True)
        
        print(f"Processed audio saved: {output_filepath}")
        
        # Return the processed audio file
        return StreamingResponse(
            open(output_filepath, "rb"), 
            media_type="audio/ogg",
            headers={"Content-Disposition": f"attachment; filename={output_filename}"}
        )
        
    except Exception as e:
        print(f"Error during stem removal: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Stem removal failed: {str(e)}")
    
    finally:
        # Clean up the uploaded file
        if os.path.exists(unique_filename):
            os.remove(unique_filename)
        
        # Clean up output directory
        if os.path.exists("output"):
            shutil.rmtree("output")
            os.makedirs("output", exist_ok=True)

@app.post("/remove_stems_youtube")
async def remove_stems_youtube(link: str,
                              vocals: bool = Form(False, description="Remove vocals from the audio"),
                              drums: bool = Form(False, description="Remove drums from the audio"),
                              bass: bool = Form(False, description="Remove bass from the audio"),
                              guitar: bool = Form(False, description="Remove guitar from the audio"),
                              piano: bool = Form(False, description="Remove piano from the audio"),
                              other: bool = Form(False, description="Remove other instruments from the audio")):
    """
    Download audio from a YouTube video and remove selected stems.
    
    Uses htdemucs_6s model to separate into 6 stems and then removes the selected ones:
    - Vocals
    - Drums  
    - Bass
    - Guitar
    - Piano
    - Other
    
    Returns OGG audio file with the selected stems removed.
    """
    try:
        # Download the YouTube audio
        print(f"Downloading audio from YouTube: {link}")
        filename = download_youtube_audio(link)
        
        if not os.path.exists(filename):
            raise HTTPException(status_code=500, detail="Failed to download YouTube audio")
        
        print(f"Downloaded audio file: {filename}")
        
        # Get the filename without the path for the UploadFile object
        audio_filename = os.path.basename(str(filename))
        
        # Call the remove_stems function with the downloaded file
        with open(filename, 'rb') as f:
            # Create an UploadFile object from the downloaded file with proper filename
            upload_file = UploadFile(filename=audio_filename, file=f)
            result = await remove_stems(upload_file, vocals, drums, bass, guitar, piano, other)
        
        # Clean up the downloaded file
        if os.path.exists(filename):
            os.remove(filename)
            print(f"Cleaned up downloaded file: {filename}")
        
        return result
        
    except Exception as e:
        print(f"Error during YouTube stem removal: {str(e)}")
        # Clean up any downloaded files
        if 'filename' in locals() and os.path.exists(filename):
            os.remove(filename)
        raise HTTPException(status_code=500, detail=f"YouTube stem removal failed: {str(e)}")


@app.post("/separate_youtube_audio")
async def separate_youtube_audio_6_stems(link: str):
    """
    Download audio from a YouTube video and separate it into 6 stems using htdemucs_6s model:
    - Vocals
    - Drums
    - Bass
    - Guitar
    - Piano
    - Other
    
    Returns a ZIP file containing all 6 stems as OGG files (high quality).
    """
    try:
        # Download the YouTube audio
        print(f"Downloading audio from YouTube: {link}")
        filename = download_youtube_audio(link)
        
        if not os.path.exists(filename):
            raise HTTPException(status_code=500, detail="Failed to download YouTube audio")
        
        print(f"Downloaded audio file: {filename}")
        
        # Get the filename without the path for the UploadFile object
        audio_filename = os.path.basename(str(filename))
        
        # Call the separate_6_stems function with the downloaded file
        with open(filename, 'rb') as f:
            # Create an UploadFile object from the downloaded file with proper filename
            upload_file = UploadFile(filename=audio_filename, file=f)
            result = await separate_6_stems(upload_file)
        
        # Clean up the downloaded file
        if os.path.exists(filename):
            os.remove(filename)
            print(f"Cleaned up downloaded file: {filename}")
        
        return result
        
    except Exception as e:
        print(f"Error during YouTube audio separation: {str(e)}")
        # Clean up any downloaded files
        if 'filename' in locals() and os.path.exists(filename):
            os.remove(filename)
        raise HTTPException(status_code=500, detail=f"YouTube audio separation failed: {str(e)}")


def convert_mono_to_stereo(input_path: str, output_path: str):
    command = ['ffmpeg', '-i', input_path, '-ac', '2', output_path]
    subprocess.run(command, check=True)

def merge_audio(instrumental_path: str, vocal_path: str, output_path: str):
    command = [
        'ffmpeg', '-i', instrumental_path, '-i', vocal_path, '-filter_complex',
        '[0:a][1:a]amix=inputs=2:duration=first:dropout_transition=2,volume=4dB[out]',
        '-map', '[out]', '-c:a', 'aac', '-b:a', '128k', output_path
    ]
    subprocess.run(command, check=True)

def merge_multiple_audio(audio_paths: list = None, output_path: str = None, instrumental_path: str = None, vocal_path: str = None):
    # Combine all audio paths
    all_paths = []
    if audio_paths:
        # Prepend output/ to each path in audio_paths
        all_paths.extend([os.path.join("output", path) for path in audio_paths])
    if instrumental_path:
        all_paths.append(instrumental_path)
    if vocal_path:
        all_paths.append(vocal_path)
        
    if not all_paths:
        raise ValueError("No audio files provided")
    if not output_path:
        raise ValueError("Output path is required")
    
    # Build input arguments
    input_args = []
    for path in all_paths:
        input_args.extend(['-i', path])
    
    # Build filter string
    inputs_count = len(all_paths)
    filter_inputs = ''.join(f'[{i}:a]' for i in range(inputs_count))
    filter_complex = f'{filter_inputs}amix=inputs={inputs_count}:duration=first:dropout_transition=2,volume=10dB[out]'
    
    command = [
        'ffmpeg',
        *input_args,
        '-filter_complex', filter_complex,
        '-map', '[out]',
        '-c:a', 'aac',
        '-b:a', '128k',
        output_path
    ]
    
    try:
        subprocess.run(command, check=True)
        # Clean up audio_paths files after successful merge
        if audio_paths:
            for path in audio_paths:
                full_path = os.path.join("output", path)
                if os.path.exists(full_path):
                    os.remove(full_path)
    except subprocess.CalledProcessError as e:
        raise e
    
def extract_youtube_video_id(url):
    # Regex pattern to match YouTube video IDs from various URL formats
    pattern = r'(?:youtube\.com\/(?:[^\/\n\s]+\/\S+\/|(?:v|e(?:mbed)?)\/|\S*?[?&]v=)|youtu\.be\/)([a-zA-Z0-9_-]{11})'
    match = re.search(pattern, url)
    if match:
        return match.group(1)
    else:
        return None

@app.post("/sing_audio")
async def sing_audio(file: UploadFile = File(..., description="The audio file to process."),
                    model_name: str = Form("AyanaAoba.pth", description="Name of the voice model to use."),
                    index_path: str = Form("added_IVF256_Flat_nprobe_1_AyanaAoba_v2.index", description="Path to the index file of the voice model."),
                    f0up_key: int = Form(7, description="Transpose (integer, number of semitones, raise by an octave: 12, lower by an octave: -12)."),
                    f0method: str = Form("rmvpe", description="Select the pitch extraction algorithm ('pm': faster extraction but lower-quality speech; 'harvest': better bass but extremely slow; 'crepe': better quality but GPU intensive), 'rmvpe': best quality, and little GPU requirement"),
                    index_rate: float = Form(0.76, description="Search feature ratio (controls accent strength, too high has artifacting)"),
                    device: str = Form("cuda", description="Select devices used to infer the model ('cpu', 'cuda', 'cuda:0', 'cuda:1', etc.)"),
                    is_half: bool = Form(False, description="Whether to use half precision."),
                    filter_radius: int = Form(4, description="If >=3: apply median filtering to the harvested pitch results. The value represents the filter radius and can reduce breathiness."),
                    resample_sr: int = Form(0, description="Resample the output audio in post-processing to the final sample rate. Set to 0 for no resampling"),
                    rms_mix_rate: float = Form(0.6, description="Adjust the volume envelope scaling. Closer to 0, the more it mimicks the volume of the original vocals. Can help mask noise and make volume sound more natural when set relatively low. Closer to 1 will be more of a consistently loud volume."),
                    protect: float = Form(0.3, description="Protect voiceless consonants and breath sounds to prevent artifacts such as tearing in electronic music. Set to 0.5 to disable. Decrease the value to increase protection, but it may reduce indexing accuracy."),
                    multi_process_vocal: bool = Form(False, description="When multi process vocal is set to true, the UVR server will process the vocal stem further using additional model with 'UVR-MDX-NET_Crowd_HQ_1.onnx' and 'Reverb_HQ_By_FoxJoy.onnx'. This will increase the processing time. but this will clean additional reverb and crowd noise from the vocal stem. potentially making the vocal stem sound cleaner with music that has more reverb and crowd noise."),
                    no_reverb: bool = Form(False, description="When set to true, reverb and echo tracks will not be mixed into the final output, resulting in a cleaner vocal without reverb effects."),
                    enable_denoise: bool = Form(False, description="When set to true, applies DeepFilterNet3 noise reduction to the separated vocals before voice conversion. This can significantly improve the quality by removing background noise and artifacts."),
                    link: Optional[str] = Form(None, description="Link to the YouTube video to process.")):
    
    if link:
        video_id = extract_youtube_video_id(link)
        # Determine the inferMode based on multi_process_vocal value
        inferMode = "full" if multi_process_vocal else "fast"
        # Determine the no_reverb mode string
        no_reverb_mode = "no-reverb" if no_reverb else "with-reverb"
        # Construct the file name prefix to check
        file_name_prefix = f"{video_id}_{model_name}_{f0up_key}_{inferMode}_{no_reverb_mode}"
        # Save the uploaded file to disk
        filename = Path(file.filename).name
        unique_filename = str(file_name_prefix)
        with open(unique_filename, "wb") as f:
            f.write(await file.read())
    else :
        # Save the uploaded file to disk
        filename = Path(file.filename).name
        unique_filename = str(uuid.uuid4())
        with open(unique_filename, "wb") as f:
            f.write(await file.read())

    # Initialize the Separator class
    separator = Separator(output_dir="output")
    
    # Load a machine learning model
    separator.load_model(model_filename='Kim_Vocal_2.onnx')
    
    # Perform the separation on the uploaded file
    output_files = separator.separate(unique_filename)
    os.remove(unique_filename)
    print(f"Separation complete! Output file(s): {' '.join(output_files)}")

    # Find the vocal file
    output_file_vocals = next((f for f in output_files if "_(Vocals)_" in f), None)
    output_file_instrumental = next((f for f in output_files if "_(Instrumental)_" in f), None)
    if output_file_vocals is None:
        raise Exception("Vocal file not found")

    if multi_process_vocal:
        separator = Separator(output_dir="output", vr_params={"batch_size": 2, "window_size": 512, "aggression": 5})
        
        # Dictionary of model names with tuples containing (vocal_stem, reverb_stem)
        models_stems = {
            'UVR-DeEcho-DeReverb.pth': (0, 1),
            'UVR-De-Echo-Normal.pth': (0, 1),
            '5_HP-Karaoke-UVR.pth': (1, 0)
        }
        
        # Store echo/reverb stems from each phase
        echo_reverb_stems = []
        
        # Use a dedicated variable for the current vocal file being processed
        current_vocal_file = output_file_vocals
        
        for phase, (model, (vocal_stem_idx, reverb_stem_idx)) in enumerate(models_stems.items(), 1):
            print(f"############## Phase {phase}: Processing with {model}")
            separator.load_model(model_filename=model)
            
            # Ensure we have the full path for separation
            current_vocal_path = os.path.join("output", current_vocal_file)
            
            # Separate and get both stems
            separated_stems = separator.separate(current_vocal_path)
            
            # Update the vocal file for next iteration
            current_vocal_file = separated_stems[vocal_stem_idx]
            
            # Store the reverb/echo stem
            echo_reverb_stems.append(separated_stems[reverb_stem_idx])
            
            print(f"Phase {phase} complete - Vocal: {current_vocal_file}, Reverb: {separated_stems[reverb_stem_idx]}")
        
        # Update the main vocal file variable with the final cleaned result
        output_file_vocals = current_vocal_file
        print('############## FINAL VOCAL AFTER CLEANING:', output_file_vocals)
        print('############## REVERB STEMS COLLECTED:', echo_reverb_stems)

    print(output_file_vocals)
    # Convert the vocal output file to MP3
    vocal_file_path = os.path.join("output", output_file_vocals)
    instrumental_file_path = os.path.join("output", output_file_instrumental)
    if os.path.exists(vocal_file_path):
        # Apply DeepFilterNet noise reduction if enabled
        if enable_denoise:
            print("#################### Applying DeepFilterNet noise reduction to vocals...")
            denoised_vocal_path = vocal_file_path.rsplit('.', 1)[0] + '_denoised.wav'
            apply_deepfilter_noise_reduction(vocal_file_path, denoised_vocal_path)
            # Replace the vocal file with the denoised version if it was created successfully
            if os.path.exists(denoised_vocal_path):
                os.remove(vocal_file_path)
                vocal_file_path = denoised_vocal_path
                print("#################### DeepFilterNet noise reduction applied successfully!")
            else:
                print("#################### DeepFilterNet failed, continuing with original vocals")
        
        # Convert to MP3
        mp3_vocal_file_path = vocal_file_path.rsplit('.', 1)[0] + '.mp3'
        subprocess.run(['ffmpeg', '-i', vocal_file_path, mp3_vocal_file_path])
        os.remove(vocal_file_path)

    # Post the vocal file to voice2voice service
    print("#################### Posting the vocal file to voice2voice service...")
    with open(mp3_vocal_file_path, 'rb') as f:
        response = requests.post(voice2voice_endpoint, files={'input_file': f}, params={'model_name': model_name, 'index_path': index_path, 'f0up_key': f0up_key, 'f0method': f0method, 'index_rate': index_rate, 'device': device, 'is_half': is_half, 'filter_radius': filter_radius, 'resample_sr': resample_sr, 'rms_mix_rate': rms_mix_rate, 'protect': protect}, stream=True)
    converted_vocal_file_path = 'converted_' + mp3_vocal_file_path

    with open(converted_vocal_file_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)
    
    print("#################### Vocal file converted successfully!")
    print(converted_vocal_file_path)
    os.remove(mp3_vocal_file_path)

    # Boost the volume of the converted vocal file
    boosted_vocal_file_path = 'boosted_' + converted_vocal_file_path
    subprocess.run(['ffmpeg', '-i', converted_vocal_file_path, '-filter:a', 'volume=9dB', boosted_vocal_file_path])

    # Delete the converted vocal file
    os.remove(converted_vocal_file_path)

    # Specify the directory where you want to save the merged file
    output_directory = "final_output"

    # Merge the boosted vocal file with the instrumental file
    merged_file_name = unique_filename + "_merged.m4a"
    merged_file_path = os.path.join(output_directory, merged_file_name)
    # Convert the vocal file to stereo
    stereo_vocal_file_path = 'stereo_' + mp3_vocal_file_path
    convert_mono_to_stereo(boosted_vocal_file_path, stereo_vocal_file_path)

    if multi_process_vocal:
        # Merge the stereo vocal file and the reverb audio file with the instrumental file
        if no_reverb:
            # Don't merge reverb - just merge vocal and instrumental
            merge_audio(instrumental_file_path, stereo_vocal_file_path, merged_file_path)
            # Clean up reverb stems since we're not using them
            for reverb_stem in echo_reverb_stems:
                reverb_path = os.path.join("output", reverb_stem)
                if os.path.exists(reverb_path):
                    os.remove(reverb_path)
        else:
            # Merge with reverb as before
            merge_multiple_audio(audio_paths=echo_reverb_stems, instrumental_path=instrumental_file_path, vocal_path=stereo_vocal_file_path, output_path=merged_file_path)
    else:
        # Merge the stereo vocal file with the instrumental file
        merge_audio(instrumental_file_path, stereo_vocal_file_path, merged_file_path)

    # Delete the original uploaded file
    os.remove(instrumental_file_path)
    os.remove(boosted_vocal_file_path)
    os.remove(stereo_vocal_file_path)
    output_directory = "output"
    # Remove everything in the output directory
    shutil.rmtree(output_directory)
    # Recreate the output directory to ensure it exists for future operations
    os.makedirs(output_directory, exist_ok=True)

    return StreamingResponse(open(merged_file_path, "rb"), media_type="audio/mp4")

@app.post("/download_youtube_audio")
async def download_youtube(link: str):
    filename_mp3 = download_youtube_audio_and_convert(link)
    file_like = open(filename_mp3, mode="rb")
    return StreamingResponse(file_like, media_type="audio/mpeg")

def download_youtube_audio_and_convert(link: str):
    # Generate a unique filename for the downloaded video
    filename_webm = str(uuid.uuid4()) + ".webm"
    filename_mp3 = filename_webm.replace(".webm", ".mp3")

    # Run yt-dlp as a subprocess to download the video
    yt_dlp_result = subprocess.run(["yt-dlp", "--cookies", "ytcookies.txt", "--remote-components", "ejs:github", link, "-f", "ba", "-o", filename_webm], capture_output=True, text=True)

    # If yt-dlp subprocess exited with a non-zero status code, raise an HTTP exception
    if yt_dlp_result.returncode != 0:
        raise HTTPException(status_code=500, detail=yt_dlp_result.stderr)

    # Convert the downloaded video to mp3 using ffmpeg
    ffmpeg_result = subprocess.run(["ffmpeg", "-i", filename_webm, filename_mp3], capture_output=True, text=True)

    # If ffmpeg subprocess exited with a non-zero status code, raise an HTTP exception
    if ffmpeg_result.returncode != 0:
        raise HTTPException(status_code=500, detail=ffmpeg_result.stderr)

    # Return the filename of the mp3 file
    return filename_mp3

def download_youtube_audio(link: str):
    video_id = extract_youtube_video_id(link)
    download_dir = Path('download')
    download_dir.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
    filename = download_dir / f"{video_id}.webm"

    # Run yt-dlp as a subprocess to download the video
    result = subprocess.run(["yt-dlp", "--cookies", "ytcookies.txt", "--remote-components", "ejs:github", link, "-f", "ba", "-o", str(filename)], capture_output=True, text=True)

    # If the subprocess exited with a non-zero status code, raise an HTTP exception with the error output
    if result.returncode != 0:
        raise HTTPException(status_code=500, detail=result.stderr)

    return filename

def download_youtube_video(link: str):
    # Generate a unique filename for the downloaded video
    filename = str(uuid.uuid4()) + ".webm"

    # Run yt-dlp as a subprocess to download the video
    result = subprocess.run(["yt-dlp", "--cookies", "ytcookies.txt", "--remote-components", "ejs:github", link, "-o", filename], capture_output=True, text=True)

    # If the subprocess exited with a non-zero status code, raise an HTTP exception with the error output
    if result.returncode != 0:
        raise HTTPException(status_code=500, detail=result.stderr)

    return filename

@app.post("/download_youtube_video")
async def dl_youtube(link: str):
    filename = download_youtube_video(link)
    file_like = open(filename, mode="rb")
    return StreamingResponse(file_like, media_type="video/mp4")

@app.post("/download_youtube_h264")
async def download_youtube_h264(link: str, use_gpu: bool = True, max_height: int = 720, compress_level: int = 2):
    """
    Download a YouTube video and convert it to H.264 format optimized for web streaming
    
    Args:
        link: YouTube video URL
        use_gpu: Whether to use GPU acceleration for encoding (default: True)
        max_height: Maximum video height to limit file size (default: 720p)
        compress_level: Compression level (1-3, where 3 is smallest file size but lower quality)
    """
    try:
        
        video_id = link    
        # Create absolute paths for files
        current_dir = os.path.abspath(os.getcwd())
        temp_filename = os.path.join(current_dir, f"{video_id}_{uuid.uuid4()}.mp4")
        output_filename = os.path.join(current_dir, f"{video_id}_{uuid.uuid4()}.mp4")
        
        print(f"Current directory: {current_dir}")
        print(f"Temp file path: {temp_filename}")
        
        # Download the YouTube video with resolution limited to max_height
        print(f"Downloading video from YouTube: {link} (max {max_height}p)")
        try:
            # Build format string to limit resolution
            format_string = f"bestvideo[height<={max_height}][ext=mp4]+bestaudio[ext=m4a]/best[height<={max_height}]"
            
            result = subprocess.run(
                ["yt-dlp",
                 "--cookies", "ytcookies.txt",
                 "--remote-components", "ejs:github",
                 "--no-playlist",
                 "-f", format_string, 
                 "-o", temp_filename, 
                 "--force-overwrites",
                 link],
                capture_output=True, text=True, check=True
            )
            print(f"Download completed successfully")
        except subprocess.CalledProcessError as e:
            print(f"ERROR in download: {e.stderr}")
            raise HTTPException(status_code=500, detail=f"Error downloading video: {e.stderr}")
        
        # Check if the file exists and log its size
        if os.path.exists(temp_filename):
            file_size = os.path.getsize(temp_filename) / (1024 * 1024)  # Size in MB
            print(f"Downloaded file size: {file_size:.2f} MB")
        else:
            # Look for any file that might have been created by yt-dlp
            print("Looking for downloaded file in current directory...")
            for file in os.listdir(current_dir):
                if video_id in file and os.path.isfile(file):
                    print(f"Found potential file: {file}")
                    temp_filename = os.path.join(current_dir, file)
                    break
            else:
                print("No matching files found")
                raise HTTPException(status_code=500, detail="Downloaded file not found")
        
        # Convert the downloaded video to H.264 format with compression
        print(f"Converting video to H.264 with compression level {compress_level}: {temp_filename} → {output_filename}")
        try:
            convert_to_h264(temp_filename, output_filename, use_gpu=use_gpu, compress_level=compress_level)
            output_size = os.path.getsize(output_filename) / (1024 * 1024)
            print(f"Conversion successful, output size: {output_size:.2f} MB")
        except Exception as e:
            print(f"ERROR during conversion: {str(e)}")
            # If GPU conversion failed, try CPU fallback
            if use_gpu:
                print("Attempting CPU fallback...")
                try:
                    convert_to_h264(temp_filename, output_filename, use_gpu=False, compress_level=compress_level)
                    print(f"CPU fallback conversion successful")
                except Exception as e2:
                    print(f"CPU fallback also failed: {str(e2)}")
                    raise HTTPException(status_code=500, detail=f"Video conversion failed: {str(e)} (CPU fallback also failed)")
            else:
                raise HTTPException(status_code=500, detail=f"Video conversion failed: {str(e)}")
        
        # Delete the temporary file
        try:
            os.remove(temp_filename)
            print(f"Temporary file removed: {temp_filename}")
        except Exception as e:
            print(f"Warning: Failed to delete temporary file: {str(e)}")
        
        # Return the converted video as a FileResponse instead of StreamingResponse for better reliability
        print(f"Serving file: {output_filename}")
        return FileResponse(
            path=output_filename,
            media_type="video/mp4",
            filename=f"{video_id}.mp4"
        )
        
    except Exception as e:
        print(f"CRITICAL ERROR in download_youtube_h264: {str(e)}")
        import traceback
        print(traceback.format_exc())
        
        # Clean up any temporary files that might have been created
        try:
            if 'temp_filename' in locals() and os.path.exists(temp_filename):
                os.remove(temp_filename)
                print(f"Cleaned up temp file: {temp_filename}")
            if 'output_filename' in locals() and os.path.exists(output_filename):
                os.remove(output_filename)
                print(f"Cleaned up output file: {output_filename}")
        except Exception as cleanup_error:
            print(f"Error during cleanup: {str(cleanup_error)}")
            
        raise HTTPException(status_code=500, detail=str(e))

def convert_to_h264(input_path: str, output_path: str, use_gpu: bool = True, compress_level: int = 2):
    """
    Convert video to H.264 format using FFmpeg with settings optimized for web streaming and smaller file size
    
    Args:
        input_path: Path to input video file
        output_path: Path for output file
        use_gpu: Whether to attempt NVIDIA GPU acceleration
        compress_level: Compression level (1-3, where 3 is smallest file size but lower quality)
    """
    # Check input file exists
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # Define compression settings based on level
    if compress_level == 1:
        # Light compression
        maxrate = '5M'
        crf_value = '26'     # Lower CRF = higher quality
        preset = 'fast' if not use_gpu else 'p4'
    elif compress_level == 2:
        # Medium compression (default)
        maxrate = '3M'
        crf_value = '30'
        preset = 'fast' if not use_gpu else 'p4'
    else:
        # Heavy compression
        maxrate = '2M'
        crf_value = '32'
        preset = 'veryfast' if not use_gpu else 'p5'
    
    # Get video height to determine if scaling is needed
    probe_cmd = ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 
                 'stream=height', '-of', 'csv=s=x:p=0', input_path]
    try:
        height = int(subprocess.check_output(probe_cmd).decode().strip())
        print(f"Original video height: {height}px")
        
        # Only scale if height > max_height
        if height > 720:
            # Use simpler scaling filter that works across FFmpeg versions
            scale_filter = 'scale=-2:720'
            print(f"Scaling video to 720p")
        else:
            # No scaling needed
            scale_filter = None
            print(f"No scaling needed, keeping original resolution")
    except Exception as e:
        print(f"Could not determine video height: {str(e)}")
        scale_filter = 'scale=-2:720'  # Default to 720p to be safe
    
    # Base command components
    if scale_filter:
        base_command = ['ffmpeg', '-i', input_path, '-vf', scale_filter]
    else:
        base_command = ['ffmpeg', '-i', input_path]
    
    if use_gpu:
        try:
            # Check if NVENC is available
            print("Checking for NVIDIA encoder availability...")
            nvidia_check = subprocess.run(
                ['ffmpeg', '-hide_banner', '-encoders'], 
                capture_output=True, text=True
            )
            
            if 'h264_nvenc' in nvidia_check.stdout:
                # Use NVIDIA hardware acceleration with optimized settings
                output_options = [
                    '-c:v', 'h264_nvenc',        # Use NVIDIA GPU encoder
                    '-preset', preset,           # Encoding preset
                    '-tune', 'hq',               # Optimize for visual quality
                    '-rc:v', 'vbr',              # Variable bitrate mode
                    '-cq:v', crf_value,          # Quality control
                    '-maxrate', maxrate,         # Limit maximum bitrate
                    '-bufsize', '8M',            # Encoding buffer size
                    '-c:a', 'aac',               # AAC audio codec
                    '-b:a', '96k',               # Reduced audio bitrate
                    '-movflags', '+faststart'    # Web streaming optimization
                ]
                print("Using NVIDIA GPU acceleration for video encoding")
            else:
                raise Exception("NVENC not available")
        except Exception as e:
            print(f"GPU encoding not available, falling back to CPU: {str(e)}")
            output_options = get_cpu_encoding_options(crf_value, maxrate, preset)
    else:
        print("Using CPU encoding as requested")
        output_options = get_cpu_encoding_options(crf_value, maxrate, preset)
    
    # Run FFmpeg command with progress monitoring
    print(f"Running FFmpeg conversion with compression level {compress_level}...")
    result = subprocess.run(base_command + output_options + [output_path], capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"FFmpeg error output: {result.stderr}")
        raise Exception(f"FFmpeg error (code {result.returncode}): {result.stderr}")
    else:
        print("FFmpeg conversion completed successfully")
        return True
    
def get_cpu_encoding_options(crf_value='30', maxrate='3M', preset='fast'):
    """Get optimized CPU encoding options for H.264 with better size/speed balance"""
    return [
        '-c:v', 'libx264',        # H.264 video codec (CPU-based)
        '-crf', crf_value,        # Quality (higher = smaller files)
        '-preset', preset,        # Encoding preset
        '-maxrate', maxrate,      # Limit maximum bitrate
        '-bufsize', '8M',         # Encoding buffer size
        '-c:a', 'aac',            # AAC audio codec
        '-b:a', '96k',            # Audio bitrate (reduced)
        '-movflags', '+faststart' # Optimize for web streaming
    ]
  
@app.post("/sing_youtube")
async def sing_youtube( link: str,                   
                        model_name: str = Form("AyanaAoba.pth", description="Name of the voice model to use."),
                        index_path: str = Form("added_IVF256_Flat_nprobe_1_AyanaAoba_v2.index", description="Path to the index file of the voice model."),
                        f0up_key: int = Form(7, description="Transpose (integer, number of semitones, raise by an octave: 12, lower by an octave: -12)."),
                        f0method: str = Form("rmvpe", description="Select the pitch extraction algorithm ('pm': faster extraction but lower-quality speech; 'harvest': better bass but extremely slow; 'crepe': better quality but GPU intensive), 'rmvpe': best quality, and little GPU requirement"),
                        index_rate: float = Form(0.76, description="Search feature ratio (controls accent strength, too high has artifacting)"),
                        device: str = Form("cuda", description="Select devices used to infer the model ('cpu', 'cuda', 'cuda:0', 'cuda:1', etc.)"),
                        is_half: bool = Form(False, description="Whether to use half precision."),
                        filter_radius: int = Form(4, description="If >=3: apply median filtering to the harvested pitch results. The value represents the filter radius and can reduce breathiness."),
                        resample_sr: int = Form(0, description="Resample the output audio in post-processing to the final sample rate. Set to 0 for no resampling"),
                        rms_mix_rate: float = Form(0.6, description="Adjust the volume envelope scaling. Closer to 0, the more it mimicks the volume of the original vocals. Can help mask noise and make volume sound more natural when set relatively low. Closer to 1 will be more of a consistently loud volume."),
                        protect: float = Form(0.3, description="Protect voiceless consonants and breath sounds to prevent artifacts such as tearing in electronic music. Set to 0.5 to disable. Decrease the value to increase protection, but it may reduce indexing accuracy."),
                        multi_process_vocal: bool = Form(False, description="When multi process vocal is set to true, the UVR server will process the vocal stem further using additional model with 'UVR-MDX-NET_Crowd_HQ_1.onnx' and 'Reverb_HQ_By_FoxJoy.onnx'. This will increase the processing time. but this will clean additional reverb and crowd noise from the vocal stem. potentially making the vocal stem sound cleaner with music that has more reverb and crowd noise."),
                        no_reverb: bool = Form(False, description="When set to true, reverb and echo tracks will not be mixed into the final output, resulting in a cleaner vocal without reverb effects."),
                        enable_denoise: bool = Form(False, description="When set to true, applies DeepFilterNet3 noise reduction to the separated vocals before voice conversion. This can significantly improve the quality by removing background noise and artifacts.")):
    
    # Check if the link is available
    video_id = extract_youtube_video_id(link)

    # Determine the inferMode based on multi_process_vocal value
    inferMode = "full" if multi_process_vocal else "fast"

    # Determine the no_reverb mode string
    no_reverb_mode = "no-reverb" if no_reverb else "with-reverb"

    # Construct the file name prefix to check
    file_name_prefix = f"{video_id}_{model_name}_{f0up_key}_{inferMode}_{no_reverb_mode}"

    # Check if the video id is available on disk in the 'final_output' directory
    final_output_dir = Path('final_output')
    file_found = None
    for file in final_output_dir.iterdir():
        if file.is_file() and file.name.startswith(file_name_prefix):
            # File exists, return a streaming response
            file_found = file
            break

    if file_found:
        return FileResponse(path=file_found, media_type='audio/mpeg')
    else:
        # Download the YouTube audio
        filename = download_youtube_audio(link)


    # Call the sing_audio function with the downloaded file
    with open(filename, 'rb') as f:
        upload_file = UploadFile(filename=filename, file=f)
        return await sing_audio(upload_file, model_name, index_path, f0up_key, f0method, index_rate, device, is_half, filter_radius, resample_sr, rms_mix_rate, protect, multi_process_vocal, no_reverb, enable_denoise, link=link)

if __name__ == "__main__":
    uvicorn.run("separate:app", host="0.0.0.0", port=8100, log_level="info")

    