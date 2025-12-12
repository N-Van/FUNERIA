______________________________________________________________________

<div align="center">

# FUNERIA

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>
[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)
[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/paper/2020)

</div>

## Description

AI assisted segmentation of CT-scanned funerary urns.

## Installation

### Pip

```bash
# clone project
git clone https://github.com/YourGithubName/FUNERIA
cd FUNERIA

# [OPTIONAL] create conda environment
conda create -n FUNERIA python=3.9
conda activate FUNERIA

# install pytorch according to instructions
# https://pytorch.org/get-started/

# install requirements
pip install -r requirements.txt
```

### Conda

```bash
# clone project
git clone https://github.com/YourGithubName/FUNERIA
cd FUNERIA

# create conda environment and install dependencies
conda env create -f environment.yaml -n FUNERIA

# activate conda environment
conda activate FUNERIA
```

## How to run

1. Set the path of the tiff file of your urn in `configs/data/urn.yml`

2. Evaluate the model on your urn:

   ```sh
   # gpu for the trainer is recommended
   python src/eval.py trainer=gpu logger=mlflow
   # or make eval
   ```

3. You can inspect the run in MLFlow

```sh
cd logs/mlflow
mlflow ui
```

You can override any parameter from command line like this

```bash
python src/eval.py data.projection_number=64
python src/eval.py experiment=experiment_name.yaml
```

## How to crop your volume

A cli tool has been developed in the `src/crop_volume.py` script.

```sh
python src/crop_volume.py --help
```

## How to visualize your volumes

We recommend to use [`napari`](https://napari.org). This conda environment can
be installed with the line below:

```sh
conda env create -f napari-env.yaml -n napari-env
```

To open your tiff image

```sh
conda activate napari-env
napari your-volume.tiff
```

## License

FUNERIA is licensed under the MIT License.

```
MIT License

Copyright (c) 2021 ashleve

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
