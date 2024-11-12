<!--content-->
# Data collection for Leap Motion

Collect music-related motion data from the Leap Motion.  

## Install Leap Motion Python SDK
This module makes use of a compiled module called `leapc_cffi`. We include some pre-compiled python objects with our
Gemini installation from 5.17 onwards. Supported versions can be found [here](#pre-compiled-module-support). If you 
have the matching python version and have installed Gemini into the default location you can follow the steps below:

```
# Create and activate a virtual environment
pip install -r requirements.txt
pip install -e leapc-python-api
python examples/tracking_event_example.py
```

### Custom Install

This module assumes that you have the Leap SDK installed in the default location. If this is not the case
for you, you can use an environment variable to define the installation location. Define the environment variable
`LEAPSDK_INSTALL_LOCATION` to the path of the `LeapSDK` folder, if you have installed to a custom location or moved it 
somewhere else.

Example:
`export LEAPSDK_INSTALL_LOCATION="C:\Program Files\CustomDir\Ultraleap\LeapSDK"`

By default, this path is the following for each operating system:
- Windows: `C:/Program Files/Ultraleap/LeapSDK`
- Linux x64: `/usr/lib/ultraleap-hand-tracking-service`
- Linux ARM: `/opt/ultraleap/LeapSDK`
- Darwin: `/Applications/Ultraleap Hand Tracking.app/Contents/LeapSDK`

## Pre-Compiled Module Support

The included pre-compiled modules within our 5.17 release currently only support the following versions of python:

- Windows: Python 3.8
- Linux x64: Python 3.8
- Darwin: Python 3.8
- Linux ARM: Python 3.8, 3.9, 3.10, 3.11

Expanded pre-compiled support will be added soon. However, this does not restrict you to these versions, if you wish to 
use a different python version please follow the instructions below to compile your own module.

### Missing Compiled Module?

You might not have the correct matching compiled `leapc_cffi` module for your system, this can cause issues when importing
leap, such as: `ModuleNotFoundError: No module named 'leapc_cffi._leapc_cffi'`
If you'd like to build your own compiled module, you will still require a Gemini install and a C compiler of your 
choice. Follow the steps below:

```
# Create and activate a virtual environment
pip install -r requirements.txt
python -m build leapc-cffi
pip install leapc-cffi/dist/leapc_cffi-0.0.1.tar.gz
pip install -e leapc-python-api
python examples/tracking_event_example.py
```

## Download music!
Donwload music from [here](https://drive.google.com/drive/folders/16kNi7iGqIu3IZkfYwiuCi-JijAszmXom?usp=sharing) and unzip to a directory of your naming in base directory. 

## Data collection
Run `collect_data.py`. --input to specify input music directory, --output to specify where to save data, --random_order to randomize order of songs.
Press 'n' when you're ready to start the next song (chosen randomly). Press 'x' to exit and save data. The program will keep collecting data till you exit.

## Visualizing your data
Just a quick way to make sure your collected data is sane. Run `decode_leap.py` after setting `datafile` in `main()` to the name of the .npy file you want to visualize. It will export it to the same directory with the same filename (.mp4). Use a tool like iMovie to align the last frame of the audio with the video to check if alignment is good.
