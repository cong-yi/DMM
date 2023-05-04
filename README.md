# An Implicit Parametric Morphable Dental Model

### [Project Page](https://vcai.mpi-inf.mpg.de/projects/DMM/) | [Paper](https://arxiv.org/abs/2211.11402)

This is a PyTorch implementation of the Siggraph Asia 2022 Paper "An Implicit Parametric Morphable Dental Model".

## Installation

```
git clone https://github.com/cong-yi/DMM.git
cd DMM
conda create -n dmm python=3.9
conda activate dmm
pip install -r requirements.txt
```

## Data Layout

Our training data follows the dataset structure used in [DeepSDF](https://github.com/facebookresearch/DeepSDF) but adapt it for dental data with semantic labels and tooth centroids. The structure is as follows:

```
<data_source_name>/
    avg_centroids.txt
    SdfSamples/
        <instance_name>.npz
        <instance_name>.pkl
```

where `<instance_name>.npz` and `<instance_name>.pkl` are samples and centroids respectively.

Subsets of the unified data source can be reference using split files, which are stored in a simple JSON format. For examples, see `examples/splits/`.

## Training the network
```
python train_dmm.py -e ./examples/upper_dmm
```

## Pre-trained model
You can download the pre-trained models from [drive](https://drive.google.com/file/d/1nNyTdI59nUhmuoaT0YYG5CP1QsERJSUH/view?usp=share_link) and put it under the subfolder `examples/upper_dmm`.

## Evaluate the network
To generate the reference shapes:
```
python generate_meanshapes.py -e ./examples/upper_dmm
```

To reconstruct dental scans:
```
python reconstruct.py -e ./examples/upper_dmm -d ./test_data --iters 300 --lr 1e-3 -s examples/splits/test_split.json
```

Due to the protocol governing the usage of our clinical data, distribution to the public is not allowed. However, I have converted a publicly available [3D dental model](https://dentistry.co.uk/2022/03/11/3shape-model-maker-create-dental-models-from-3shape-trios-scans-in-minutes/) into SDF sampling data, complete with teeth numbering, for the purpose of conducting a simple test (put the [data](https://drive.google.com/file/d/18nNzqGEpRB5N6vki6jrDL6XEAf20nAqt/view?usp=sharing) under the subfolder `test_data`). This test data example, together with the reference shapes, serves as the reference for data alignment.

## Citation
If you find DMM useful for your research, please cite our
[paper](https://dl.acm.org/doi/10.1145/3550454.3555469):
```
@article{zhang2022dmm,
author = {Zhang, Congyi and Elgharib, Mohamed and Fox, Gereon and Gu, Min and Theobalt, Christian and Wang, Wenping},
title = {An Implicit Parametric Morphable Dental Model},
year = {2022},
issue_date = {December 2022},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
volume = {41},
number = {6},
issn = {0730-0301},
url = {https://doi.org/10.1145/3550454.3555469},
doi = {10.1145/3550454.3555469},
journal = {ACM Trans. Graph.},
month = {nov},
articleno = {217},
numpages = {13},
}
```

## Acknowledgement
This code repo is heavily based on [DeepSDF](https://github.com/facebookresearch/DeepSDF), [SIREN](https://github.com/vsitzmann/siren) and [DIF-Net](https://github.com/microsoft/DIF-Net). And users can train their own network on the data from the [MICCAI Challenge](https://3dteethseg.grand-challenge.org/). Thanks for these great projects.