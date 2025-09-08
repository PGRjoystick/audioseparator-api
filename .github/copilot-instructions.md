# GitHub Copilot Instructions for Audio Separator API

## Project Overview
This is a FastAPI-based audio processing server that provides AI-powered voice synthesis and audio manipulation capabilities. The server combines multiple audio processing technologies to create singing AI voices and handle YouTube media processing.

## Core Technologies & Dependencies
- **FastAPI**: Web API framework for HTTP endpoints
- **UVR (Ultimate Vocal Remover)**: Audio source separation using ML models
- **RVC (Retrieval-based Voice Conversion)**: Voice conversion technology
- **FFmpeg**: Audio/video processing and conversion
- **PyTorch**: Machine learning framework with CUDA support
- **yt-dlp**: YouTube video/audio downloading
- **audio-separator**: Python library for audio stem separation

## Project Architecture

### Main Components
1. **Audio Separation**: Uses UVR models to separate vocals from instrumentals
2. **Voice Conversion**: Converts separated vocals using RVC models
3. **Audio Merging**: Combines processed vocals with instrumentals using FFmpeg
4. **YouTube Processing**: Downloads and compresses YouTube content
5. **File Management**: Handles temporary files and cleanup operations

### Key Directories
- `output/`: Temporary audio processing files
- `final_output/`: Final merged audio files
- `download/`: Downloaded YouTube content
- `upload/`: User uploaded files
- `venv/`: Python virtual environment

## API Endpoints

### Audio Processing Endpoints
- `POST /separate_instrumental` - Extract instrumental track only
- `POST /separate_vocals` - Extract vocal track only  
- `POST /sing_audio` - Full AI voice synthesis pipeline
- `POST /sing_youtube` - AI voice synthesis from YouTube URL

### YouTube Processing Endpoints
- `POST /download_youtube_audio` - Download audio from YouTube
- `POST /download_youtube_video` - Download video from YouTube
- `POST /download_youtube_h264` - Download and compress video to H.264

## Code Patterns & Conventions

### File Handling
- Use unique filenames with UUID to prevent conflicts
- Always clean up temporary files after processing
- Use absolute paths for file operations
- Implement safe file removal with retry logic

### Audio Processing Workflow
1. **Separation**: Use UVR models (Kim_Vocal_2.onnx, UVR-MDX-NET-Inst_full_292.onnx)
2. **Conversion**: Post vocals to voice2voice service endpoint
3. **Enhancement**: Apply volume boosting and stereo conversion
4. **Merging**: Combine processed vocals with instrumentals

### Error Handling
- Use HTTPException for API errors
- Implement subprocess error checking
- Provide detailed error messages in API responses
- Clean up files even when errors occur

## Important Configuration

### Environment Variables
- `VOICE2VOICE_ENDPOINT`: External RVC service endpoint for voice conversion

### Model Files
- `Kim_Vocal_2.onnx`: Primary vocal separation model
- `UVR-MDX-NET-Inst_full_292.onnx`: Instrumental separation model
- `UVR-DeEcho-DeReverb.pth`: Echo/reverb removal (optional)
- `5_HP-Karaoke-UVR.pth`: Karaoke processing (optional)

### Audio Processing Parameters
- `f0up_key`: Pitch shifting (semitones)
- `f0method`: Pitch extraction algorithm (rmvpe recommended)
- `index_rate`: Feature search ratio for voice conversion
- `multi_process_vocal`: Enhanced vocal cleaning option

## Development Guidelines

### When Adding New Features
1. Follow existing patterns for file handling and cleanup
2. Use proper error handling with HTTPException
3. Implement proper logging for debugging
4. Consider GPU vs CPU processing options
5. Add appropriate endpoint documentation

### Audio Processing Best Practices
- Always validate input file formats
- Implement proper temporary file management
- Use appropriate FFmpeg settings for quality vs file size
- Handle both mono and stereo audio formats
- Consider processing time for user experience

### YouTube Integration
- Extract video IDs properly for caching
- Handle various YouTube URL formats
- Implement proper format selection for downloads
- Use appropriate compression settings for web delivery

## Common Use Cases

### AI Voice Cover Creation
1. User uploads audio or provides YouTube link
2. System separates vocals from instrumentals
3. Vocals are converted using RVC models
4. Final audio is merged and returned

### Audio Separation Only
1. User uploads audio file
2. System extracts either vocals or instrumentals
3. Processed audio is returned in MP3 format

### YouTube Content Processing
1. Download audio/video from YouTube URLs
2. Apply compression and format conversion
3. Return optimized files for web streaming

## Performance Considerations
- Use GPU acceleration when available (CUDA)
- Implement background processing for long tasks
- Cache processed files using video ID + parameters
- Clean up temporary files to manage disk space
- Consider batch processing for multiple files

## Security & File Management
- Generate unique filenames to prevent conflicts
- Validate file types and sizes
- Implement proper cleanup after processing
- Use safe file removal with retry mechanisms
- Protect against path traversal attacks
