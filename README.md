# PetVision Studio

A computer vision project for separating cats and dogs from the background in images. I used PyTorch to train a U-Net model and Streamlit to make a small demo app.

## Dataset and model

I used the **Oxford-IIIT Pet Dataset** with 7,349 images:

- Training: 2,944 images
- Validation: 736 images
- Test: 3,669 images

The images were resized to 256 × 256 and the segmentation masks were converted into two classes: pet and background.

The model is a small U-Net built with PyTorch. I trained each version for **5 epochs** and tried three experiments:

| Experiment | Best validation Dice |
|---|---:|
| Baseline (BCE + Dice loss) | **0.8286** |
| Data augmentation | 0.8075 |
| BCE loss only | 0.8167 |

The baseline gave the best validation result, so I used it for the final test.

## Results

On the test set, the baseline model achieved:

- **Dice: 0.8243**
- **IoU: 0.7194**

I also checked some incorrect predictions. Seven test masks were completely empty, and the model sometimes includes parts of the background in its predictions. It works better on some photos than others.

## ONNX and Streamlit

I exported the trained model to ONNX and compared CPU inference times:

| Format | Median inference time |
|---|---:|
| PyTorch | 289.8 ms |
| ONNX Runtime | 178.2 ms |

ONNX Runtime was about **1.6× faster** in my test. The model size stayed almost the same (about 7.4 MB).

The Streamlit app lets you upload an image, see the predicted mask and overlay, and save a transparent PNG.

## Run the app

Clone or download the repository, then open a terminal in the project folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Make sure `unet_baseline.onnx` and `unet_baseline.onnx.data` are in the same folder as `app.py`.

## Project files

- `notebooks/` — training and evaluation code
- `results/` — experiment tables and figures
- `app.py` — Streamlit demo
- `requirements.txt` — app dependencies
- `unet_baseline.onnx` and `.onnx.data` — exported model

## What could be improved

I only trained each experiment for five epochs. With more time, I would try longer training, more experiments and better handling of difficult backgrounds. The app is mainly a demo, not a finished background-removal tool.
