
<div align="center">    
 
# Ambiguous Datasets     

[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)
[![Conference](http://img.shields.io/badge/NeurIPS-2019-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)
[![Conference](http://img.shields.io/badge/ICLR-2019-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)
[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)  
<!--
ARXIV   
[![Paper](http://img.shields.io/badge/arxiv-math.co:1480.1111-B31B1B.svg)](https://www.nature.com/articles/nature14539)
-->
![CI testing](https://github.com/PyTorchLightning/ambiguous-dataset/workflows/CI%20testing/badge.svg?branch=master&event=push)


<!--  
Conference   
-->   
</div>
 
## Description   
This repository contains ambiguous datasets generated using a conditional variational autoencoder (CVAE) approach. The datasets contain images that are ambiguous between a pair of classes. This class interpolation is done by conditional generation through the CVAE with a class-vector and blend factor (0.5) for the desired class. Currently, only MNIST and EMNIST are supported.

## How to run   
First, install dependencies   
```bash
# clone project   
git clone https://github.com/ABL-Lab/ambiguous-dataset

# install project   
cd ambiguous-dataset 
pip install -e .   
pip install -r requirements.txt
# download ambiguous datasets (MNIST and EMNIST)
sh ambiguous/dataset/download_amnist.sh
sh ambiguous/dataset/download_aemnist.sh
```

## Importing Ambiguous Dataset to your own project
This project is setup as a package which means you can easily import any file into any other file like so.
```python
from ambiguous.dataset.dataset import *

root = 'path_to_ambiguous_dataset'
trainset = DatasetFromNPY(root=root, download=False, train=True, transform=None)
testset = DatasetFromNPY(root=root, download=False, train=False, transform=None)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
testloader = DataLoader(testset, batch_size=64, shuffle=False)
```

### Citation   
```
@article{YourName,
  title={Your Title},
  author={Your team},
  journal={Location},
  year={Year}
}
```   
