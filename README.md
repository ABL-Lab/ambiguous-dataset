
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
Ambiguous dataset generation using conditional variational autoencoder (CVAE).  

## How to run   
First, install dependencies   
```bash
# clone project   
git clone https://github.com/ABL-Lab/ambiguous-dataset

# install project   
cd ambiguous-dataset 
pip install -e .   
pip install -r requirements.txt
```

## Importing Ambiguous Dataset to your own project
This project is setup as a package which means you can now easily import any file into any other file like so. Currently only
MNIST and EMNIST (letter MNIST) are supported. <dataset>_fly means the data is generated on the fly in the data loader, using the CVAE generator.
```python
from project.dataset.dataset import *
from project.dataset.dataset import EMNIST_fly, MNIST_fly

ambiguousDataset=EMNIST_fly(root='/share/datasets',blend=0.5)
ambiguousDataLoader = DataLoader(ambiguousDataset, batch_size=64, shuffle=True)

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
