# AnyPick DraKe

## Installation

```
cd anypick_dk
pip install -e .
```

## Grounded SAM Installation

activate venv
cd into grounded_sam pip install -r requirements.txt

pretrained weights:
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

cd into GroundingDINO pip install -e .
cd into segment_anything pip install -e .

if run into segment anything import error try: pip install "git+https://github.com/facebookresearch/segment-anything.git"
