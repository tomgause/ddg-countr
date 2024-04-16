# DotDotGoose CounTR Add-on



This repository contains scripts for an experimental machine learning add-on to the DDG object counting tool that enables CounTR driven auto-labeling.



## Requirements



Linux or WSL

CUDA device (tested on NVIDIA 3090)



## Getting Started



Before getting started with this package, you will need to set up the necessary libraries to run CounTR and DINOv on your local machine. Follow the installation steps listed in the official CounTR and DINOv Githubs. We recommend using a virtual environment for both libraries.



*CounTR notes: Skip the `FSC147` and `CARPK` dataset downloads as we won't be training or evaluating the model. Instead, download the `FSC147` fine-tuned weights.*



*DINOv notes: Run the installation steps and download the `swinL` pre-trained backbone. Then, modify the end of `task_openset` function under `demo/openset_task.py` with this code:*
``` bash
masks  = [(pred_masks_pos>0.0).cpu().numpy() for  pred_masks_pos  in  pred_masks_poses]
return  res, masks
```

To simplify the setup process, we've opted to load CounTR and DINOv libraries through the path environment variable. Before continuing, modify the `config` file with the path to your libraries, data, and fine-tuned weights.



## Inference *(WIP)*



*All instructions from this point forward are configured for the custom African birds dataset. General use will require code modifications.* To achieve optimal CounTR performance, we must provide some reference boxes around objects of interest for each image. These can be manually drawn, but this labeling effort probably defeats the purpose of automated counting. Instead, we've configured `DINOv` to autogenerate reference boxes using a set of 8 manually labeled images.





### DINOv: Box Generation



Activate your `DINOv` venv and execute `dinov_main.py` to generate results for the birds dataset.



### CounTR: Evaluation Mode



To perform inference on a large dataset for evaluation, use the evaluation mode.



```bash

python  countr_main.py  --input eval  --json path-to-dinov-results-json

```

Then, use the `eval.py` script to evaluate predictions against ground truth results.

```bash

python  eval.py  --input path-to-dinov-results-dir

```



### CounTR: Single Inference Mode



For general purposes, use the single-inference mode and specify the path to your input image in addition to any stringified input boxes in `[[[x1,x2],[y1,y2]],...]` format.



```bash

python  countr_main.py  --input  path-to-input-image  --boxes  "boxes-as-string"

```



Example input:



```bash

python  countr_main.py  --input  data/aerial-seabirds-west-africa/birds.jpg  --boxes  "[[[166, 22],[189, 46]],[[254, 144],[273, 159]],[[339, 205],[370, 221]]]"

```
