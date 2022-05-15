from setuptools import setup, find_packages

setup(
    name="stas",
    version="0.0.1",
    packages=find_packages(),
    license="MIT",
    description="stas",
    install_requires=[
        "pytorch-model-utils @ git+https://github.com/bhbbbbb/pytorch-model-utils@v0.1.4",
        "semseg @ git+https://github.com/sithu31296/semantic-segmentation",
    ],
)
