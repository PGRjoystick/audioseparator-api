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

app = FastAPI()
load_dotenv()
voice2voice_endpoint = os.getenv('VOICE2VOICE_ENDPOINT')

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

def convert_mono_to_stereo(input_path: str, output_path: str):
    command = ['ffmpeg', '-i', input_path, '-ac', '2', output_path]
    subprocess.run(command, check=True)

def merge_audio(instrumental_path: str, vocal_path: str, output_path: str):
    command = [
        'ffmpeg', '-i', instrumental_path, '-i', vocal_path, '-filter_complex',
        '[0:a][1:a]amix=inputs=2:duration=first:dropout_transition=2,volume=4dB[out]',
        '-map', '[out]', '-b:a', '128k', '-f', 'opus', output_path
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
        '-b:a', '128k',
        '-f', 'opus',
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
                    link: Optional[str] = Form(None, description="Link to the YouTube video to process.")):
    
    if link:
        video_id = extract_youtube_video_id(link)
        # Determine the inferMode based on multi_process_vocal value
        inferMode = "full" if multi_process_vocal else "fast"
        # Construct the file name prefix to check
        file_name_prefix = f"{video_id}_{model_name}_{f0up_key}_{inferMode}"
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
        
        for model, (vocal_stem, reverb_stem) in models_stems.items():
            separator.load_model(model_filename=model)
            output_file_vocals = os.path.join("output", output_file_vocals)
            
            # Separate and get both stems
            separated_stems = separator.separate(output_file_vocals)
            output_file_vocals = separated_stems[vocal_stem]
            echo_reverb_stems.append(separated_stems[reverb_stem])
        print('############## REVERB STEMS PATH ', echo_reverb_stems)

    print(output_file_vocals)
    # Convert the vocal output file to MP3
    vocal_file_path = os.path.join("output", output_file_vocals)
    instrumental_file_path = os.path.join("output", output_file_instrumental)
    if os.path.exists(vocal_file_path):
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
    merged_file_name = unique_filename + "_merged.mp3"
    merged_file_path = os.path.join(output_directory, merged_file_name)
    # Convert the vocal file to stereo
    stereo_vocal_file_path = 'stereo_' + mp3_vocal_file_path
    convert_mono_to_stereo(boosted_vocal_file_path, stereo_vocal_file_path)

    if multi_process_vocal:
        # Merge the stereo vocal file and the reverb audio file with the instrumental file
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

    return StreamingResponse(open(merged_file_path, "rb"), media_type="audio/mpeg")

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
    yt_dlp_result = subprocess.run(["yt-dlp", link, "-f", "ba", "-o", filename_webm], capture_output=True, text=True)

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
    result = subprocess.run(["yt-dlp", link, "-f", "ba", "-o", str(filename)], capture_output=True, text=True)

    # If the subprocess exited with a non-zero status code, raise an HTTP exception with the error output
    if result.returncode != 0:
        raise HTTPException(status_code=500, detail=result.stderr)

    return filename

def download_youtube_video(link: str):
    # Generate a unique filename for the downloaded video
    filename = str(uuid.uuid4()) + ".webm"

    # Run yt-dlp as a subprocess to download the video
    result = subprocess.run(["yt-dlp", link, "-o", filename, ], capture_output=True, text=True)

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
async def download_youtube_h264(link: str, use_gpu: bool = True):
    """
    Download a YouTube video and convert it to H.264 format optimized for web streaming
    
    Args:
        link: YouTube video URL
        use_gpu: Whether to use GPU acceleration for encoding (default: True)
    """
    try:
        # Generate unique filenames for temporary and output files
        video_id = extract_youtube_video_id(link)
        if not video_id:
            raise HTTPException(status_code=400, detail="Invalid YouTube URL")
            
        # Create absolute paths for files
        current_dir = os.path.abspath(os.getcwd())
        temp_filename = os.path.join(current_dir, f"{video_id}_{uuid.uuid4()}.mp4")
        output_filename = os.path.join(current_dir, f"{video_id}_{uuid.uuid4()}.mp4")
        
        print(f"Current directory: {current_dir}")
        print(f"Temp file path: {temp_filename}")
        
        # Download the YouTube video with best quality - explicitly specify output format
        print(f"Downloading video from YouTube: {link}")
        try:
            # Use -o to force specific output location and name
            result = subprocess.run(
                ["yt-dlp", 
                 "--no-playlist",
                 "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best", 
                 "-o", temp_filename, 
                 "--force-overwrites",
                 link],
                capture_output=True, text=True, check=True
            )
            print(f"yt-dlp stdout: {result.stdout}")
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
        
        # Convert the downloaded video to H.264 format
        print(f"Converting video to H.264: {temp_filename} → {output_filename}")
        try:
            convert_to_h264(temp_filename, output_filename, use_gpu=use_gpu)
            print(f"Conversion successful, output size: {os.path.getsize(output_filename) / (1024 * 1024):.2f} MB")
        except Exception as e:
            print(f"ERROR during conversion: {str(e)}")
            # If GPU conversion failed, try CPU fallback
            if use_gpu:
                print("Attempting CPU fallback...")
                try:
                    convert_to_h264(temp_filename, output_filename, use_gpu=False)
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
        
        # Return the converted video as a streaming response
        print(f"Serving file: {output_filename}")
        return StreamingResponse(
            file_streamer(output_filename), 
            media_type="video/mp4",
            headers={"Content-Disposition": f"attachment; filename=\"{video_id}.mp4\""}
        )
        
    except Exception as e:
        print(f"CRITICAL ERROR in download_youtube_h264: {str(e)}")
        # Print the full traceback for better debugging
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

def convert_to_h264(input_path: str, output_path: str, use_gpu: bool = True):
    """
    Convert video to H.264 format using FFmpeg with settings optimized for web streaming
    
    Args:
        input_path: Path to input video file
        output_path: Path for output file
        use_gpu: Whether to attempt NVIDIA GPU acceleration
    """
    # Base command components
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
                # Use NVIDIA hardware acceleration
                output_options = [
                    '-c:v', 'h264_nvenc',     # Use NVIDIA GPU encoder
                    '-preset', 'p4',          # Encoding preset (p1-p7, higher = better quality)
                    '-tune', 'hq',            # Optimize for visual quality
                    '-rc:v', 'vbr',           # Variable bitrate mode
                    '-cq:v', '23',            # Constrained quality level (lower = better)
                    '-b:v', '0',              # Let quality control determine bitrate
                    '-c:a', 'aac',            # AAC audio codec
                    '-b:a', '128k',           # Audio bitrate
                    '-movflags', '+faststart' # Web streaming optimization
                ]
                print("Using NVIDIA GPU acceleration for video encoding")
            else:
                # Fall back to CPU encoding
                print("NVENC not found in FFmpeg encoders")
                raise Exception("NVENC not available")
        except Exception as e:
            print(f"GPU encoding not available, falling back to CPU: {str(e)}")
            output_options = get_cpu_encoding_options()
    else:
        # Use CPU encoding
        print("Using CPU encoding as requested")
        output_options = get_cpu_encoding_options()
    
    # Run FFmpeg command
    print(f"Running FFmpeg with command: {' '.join(base_command + output_options + [output_path])}")
    result = subprocess.run(base_command + output_options + [output_path], capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"FFmpeg error output: {result.stderr}")
        raise Exception(f"FFmpeg error (code {result.returncode}): {result.stderr}")
    else:
        print("FFmpeg conversion completed successfully")

def get_cpu_encoding_options():
    """Get the standard CPU encoding options for H.264"""
    return [
        '-c:v', 'libx264',        # H.264 video codec (CPU-based)
        '-crf', '23',             # Quality (lower = better)
        '-preset', 'medium',      # Encoding speed/quality balance
        '-c:a', 'aac',            # AAC audio codec
        '-b:a', '128k',           # Audio bitrate
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
                        multi_process_vocal: bool = Form(False, description="When multi process vocal is set to true, the UVR server will process the vocal stem further using additional model with 'UVR-MDX-NET_Crowd_HQ_1.onnx' and 'Reverb_HQ_By_FoxJoy.onnx'. This will increase the processing time. but this will clean additional reverb and crowd noise from the vocal stem. potentially making the vocal stem sound cleaner with music that has more reverb and crowd noise.")):
    
    # Check if the link is available
    video_id = extract_youtube_video_id(link)

    # Determine the inferMode based on multi_process_vocal value
    inferMode = "full" if multi_process_vocal else "fast"

    # Construct the file name prefix to check
    file_name_prefix = f"{video_id}_{model_name}_{f0up_key}_{inferMode}"

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
        return await sing_audio(upload_file, model_name, index_path, f0up_key, f0method, index_rate, device, is_half, filter_radius, resample_sr, rms_mix_rate, protect, multi_process_vocal, link=link)

if __name__ == "__main__":
    uvicorn.run("separate:app", host="0.0.0.0", port=8100, log_level="info")