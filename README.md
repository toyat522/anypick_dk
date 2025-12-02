# AnyPick DraKe

## Installation

```
cd anypick_dk
pip install -e .
```

### Grounded SAM Installation

```
cd external/grounded_sam
pip install -r requirements.txt
```

```
cd GroundingDINO
pip install -e .
```

```
cd ../segment_anything
pip install -e .
```

Download pretrained weights:

```
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

> NOTE: If you run into Segment Anything import error, try: `pip install "git+https://github.com/facebookresearch/segment-anything.git"`