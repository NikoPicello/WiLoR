This implementation is based on **[WiLoR](https://rolpotamias.github.io/WiLoR/)**, a state-of-the-art hand localization and reconstruction model;

## Installation
### Original Installation
```
git clone --recursive https://github.com/rolpotamias/WiLoR.git
cd WiLoR
```

The code has been tested with PyTorch 2.0.0 and CUDA 11.7. It is suggested to use an anaconda environment to install the the required dependencies:
```bash
conda create --name wilor python=3.10
conda activate wilor

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu117
# Install requirements
pip install -r requirements.txt
```
Download the pretrained models using:
```bash
wget https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main/pretrained_models/detector.pt -P ./pretrained_models/
wget https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main/pretrained_models/wilor_final.ckpt -P ./pretrained_models/
```
It is also required to download MANO model from [MANO website](https://mano.is.tue.mpg.de).
Create an account by clicking Sign Up and download the models (mano_v*_*.zip). Unzip and place the right hand model `MANO_RIGHT.pkl` under the `mano_data/` folder.
Note that MANO model falls under the [MANO license](https://mano.is.tue.mpg.de/license.html).

## How to use [standalone]. 
Given the pipeline, we defined the following structure for the project:
```bash 
├── resources
|   ├── sessions
|   │   ├── 000000 
|   │   │   ├── talk
|   │   │   ├── lego
|   │   │   ├── ghost
|   │   │   └── ...
|   │   ├── 000010 
|   │   |   ├── talk
|   │   |   ├── lego
|   │   |   ├── ghost
|   │   |   └── ...
|   |   └── ... 
|   ├── wilor_results
|   │   ├── 000000 
|   │   │   ├── talk
|   |   |   |   ├── cam1_wilor.pkl
|   |   |   |   ├── cam2_wilor.pkl
|   |   |   |   └── ...
|   │   │   ├── lego
|   |   |   |   ├── cam1_wilor.pkl
|   |   |   |   ├── cam2_wilor.pkl
|   |   |   |   └── ...
|   │   │   └── ...
|   │   ├── 000010 
|   │   |   ├── talk
|   |   |   |   ├── cam1_wilor.pkl
|   |   |   |   ├── cam2_wilor.pkl
|   |   |   |   └── ...
|   │   │   ├── lego
|   |   |   |   ├── cam1_wilor.pkl
|   |   |   |   ├── cam2_wilor.pkl
|   |   |   |   └── ...
|   │   |   └── ...
|   |   └── ... 
└── scripts
    ├── WiLoR
    │   ├── wilor_pipeline.py 
    |   └── ...
    ├── 3DDFA-V3
    |   └── ...
    └── ...
```

In order to run the pipeline on the sessions saved with the same format shown above just run the script:
```bash
python3 wilor_pipeline.py -b 8 --max_frames 50 --sid 000000 --aid talk --vis
```

There are some flags that can be used to specifiy:
- batch size,
- max number of frames to process,
- session id,
- activity id,
- whether to generate an output video of the detections or not. 
      

