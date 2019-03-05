# Installation

## Requirements:
- conda
- python 3.5
- pytorch 1.0
- cython
- yacs
- scikit-image
- matplotlib (for visualization)
- opencv


## Step-by-step installation:

```
conda create --name afm python=3.5
conda activate afm

pip install -r requirements.txt

conda install pytorch torchvision cudatoolkit=9.0 -c pytorch

cd lib
make
conda develop . ./lib
```