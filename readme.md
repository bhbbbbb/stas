<a href="https://colab.research.google.com/github/bhbbbbb/stas/blob/master/stas.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
</div>

## Installation

- semantic-segmentation

```sh
git clone https://github.com/sithu31296/semantic-segmentation
cd semantic-segmentaion
pip install -e .
```

- stas

```sh
git clone https://github.com/bhbbbbb/stas
cd stas
pip install -e .
```

or without scripts

```sh
pip install git+https://github.com/bhbbbbb/stas
```


- download [pretrained weights](https://drive.google.com/drive/folders/1b7bwrInTW4VLEm27YawHOAMSMikga2Ia)


## How to

### Preprocessing

- split dataset and make masks (may need to set `DATASET_ROOT` in `tools/preprocessing.py` manually)

```sh
python tools/processing.py
```

- or use `tools.preprocessing.make_mask`

## Dependency

- [semantic-segmentation](https://github.com/sithu31296/semantic-segmentation)

- [pytorch-model-utils](https://github.com/bhbbbbb/pytorch-model-utils)