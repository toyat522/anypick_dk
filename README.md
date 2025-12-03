# AnyPick DraKe

## Installation

### AnyPick DraKe Package Installation

```
cd anypick_dk
pip install -e .
```

### Grounded SAM Installation

Install dependencies and SAM:
```
cd external/grounded_sam
pip install -r requirements.txt
pip install "git+https://github.com/facebookresearch/segment-anything.git"
```

Download pretrained weights:
```
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

Install Grounding DINO:
```
cd GroundingDINO
pip install -e .
```

### Other Dependencies

```
pip install py-trees
```