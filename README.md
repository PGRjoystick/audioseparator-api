# AudioSeparator-API

AudioSeparator-API is a simple yet powerful HTTP API designed to facilitate audio separation tasks, leveraging the UVR developed by Anjok07. This project aims to provide an accessible interface for users to perform audio separation with high efficiency and accuracy.

This project also has a singing AI voice generation feature that utilizes the Retrieval-based Voice Conversion (RVC) technology developed by the RVC-Project team. This feature allows users to generate a singing AI voice from a YouTube video or an audio file.

by utilizing both the UVR and RVC technologies, AudioSeparator-API offers a comprehensive solution for audio separation and clean singing AI voice generation tasks.

## HTTP API Endpoints

The API consists of several endpoints, each tailored to handle different aspects of audio separation. Below is an overview of the available endpoints:

- `/separate_instrumental`: Separate and output instrumental stem from an audio file.
- `/separate_vocals`: Separate and output vocal stem from an audio file.
- `/sing_youtube`: Generate a singing AI voice from a YouTube video using RVC Server
- `/sing_audio`: Generate a singing AI voice from an audio file using RVC Server
- `/download_youtube_audio`: Download audio from a YouTube video
- `/download_youtube_video`: Download a YouTube video

### Additional setup for singing AI voice endpoint

#### `/sing_youtube` and `/sing_audio`

To utilize these endpoints, you must have an RVC server operational, either hosted locally or remotely. For those interested in setting up their own RVC server locally, the following steps are required:

1. Clone the RVC repository: `https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI` and setup the project
2. Manually merge the pull request from `https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/pull/1614`
3. Run `rvc_fastapi.py` to start the server.
4. rename the `.env-example` into `.env` and change your host and port of the RVC API server

This setup is necessary because the pull request has not been merged into the main RVC repository yet. However, it has been tested and confirmed to work perfectly.

## Installation and Setup

To install and set up AudioSeparator-API, ensure the following requirements are met:

- A CUDA-capable GPU with at least 6GB VRAM (Note: AMD GPUs are not supported at this time).
- [Python 3.10](https://www.python.org/downloads/release/python-3100/) and `pip` installed
- [NVIDIA CUDA Toolkit 11.8](https://developer.nvidia.com/cuda-11-8-0-download-archive) installed
- [NVIDIA cuDNN](https://developer.nvidia.com/cudnn) installed
- [FFmpeg](https://www.gyan.dev/ffmpeg/builds/) and [yt-dlp](https://github.com/yt-dlp/yt-dlp) Installed and added to `PATH`
- Run the `start.bat` file to complete the setup.

### Compatibility

Currently, AudioSeparator-API has been tested on Windows 10 with an RTX 3060Ti 8GB GPU and Python 3.10. While it has not been tested on other platforms or Python versions, it is possible to modify the dependencies to accommodate other setups. At this time, setup scripts are provided for Windows only.

## Contributing

Contributions to AudioSeparator-API are welcome! Whether it's through submitting pull requests, reporting bugs, or suggesting enhancements, your input is valuable to us.

## License

This project is licensed under the MIT [License](LICENSE).

- **Please Note:** If you choose to integrate this project into some other project using the default model or any other model trained as part of the [UVR](https://github.com/Anjok07/ultimatevocalremovergui) and [RVC](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) project, please honor the MIT license by providing credit to UVR and RVC and its developers!

## Acknowledgments

- [Anjok07](https://github.com/Anjok07) - Author of [Ultimate Vocal Remover GUI](https://github.com/Anjok07/ultimatevocalremovergui), which almost all of the code in [audioseparator](https://github.com/karaokenerds/python-audio-separator) library was copied from! Definitely deserving of credit for anything good from this project. Thank you!
- [beaverdb](https://github.com/beveradb) a.k.a [karaokenerds](https://github.com/karaokenerds) for developing the [audioseparator](https://github.com/karaokenerds/python-audio-separator) library. thank you!
- Gratitude to the RVC-Project team for their work on the [Retrieval-based Voice Conversion WebUI](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI).
- [Kuielab & Woosung Choi](https://github.com/kuielab) - Developed the original MDX-Net AI code.
- [KimberleyJSN](https://github.com/KimberleyJensen) - Advised and aided the implementation of the training scripts for MDX-Net and Demucs. Thank you!
