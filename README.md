# DotDotGoose CounTR Add-on

This repository contains scripts for an experimental machine learning add-on to the DDG object counting tool that enables CounTR driven auto-labeling.

## Requirements  

Linux or WSL  
CUDA device (tested on NVIDIA 3090)

## Getting Started  

Before getting started with this package, you will need to set up the necessary libraries to run CounTR on your local machine. Follow the installation steps listed in the official CounTR Github. We recommend using a `venv` or `conda`. Skip the FSC147 and CARPK dataset downloads as we won't be training or evaluating the model. Instead, download the FSC147 fine-tuned weights. 

To simplify the setup process, we've opted to load CounTR libraries through the path environment variable. Before continuing, modify the `config` file with the path to your CounTR library and fine-tuned weights.  

The `venv` containing the CounTR libraries should contain all the necessary packages to execute the scripts here.

## Single Inference Mode  

For general purposes, use the single-inference mode and specify the path to your input image in addition to any stringified input boxes in `[[[x1,x2],[y1,y2]],...]` format.

```bash
python main.py --input path-to-input-image --boxes "boxes-as-string"
```

Example input:

```bash
python main.py --input data/birds.jpg --boxes "[[[166, 22],[189, 46]],[[254, 144],[273, 159]],[[339, 205],[370, 221]]]"
```

## Evaluation Mode

TODO