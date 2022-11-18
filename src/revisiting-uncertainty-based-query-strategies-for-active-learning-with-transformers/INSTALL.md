# Installation Instructions

First checkout the code:
```
git clone https://github.com/webis-de/acl22-revisiting-uncertainty-based-query-strategies-for-active-learning-with-transformers.git .
```

For the remainder of this installation, the working directory is assumed to be this checked-out folder: 

```
cd acl22-revisiting-uncertainty-based-query-strategies-for-active-learning-with-transformers
```

## Installing Python Dependencies

We recommend to start with a clean python environment. 
You can get a new virtual environment at the current location as follows:
```
python -m venv venv
source venv/bin/activate
```

The python dependencies can then be installed via pip:
```
pip install -r requirements.txt
```

If your environment is compatible with the pinned pytorch and torchtext dependencies (pytorch 1.8.0 CUDA 11.1 / torchtext 0.9.0)
your installation is finished, otherwise see the next section on how to proceed.

## Manually installing PyTorch and torchtext: 

One possible obstacle could be that our pinned PyTorch / torchtext version are not the right ones for you, 
because they depend on the CUDA version of your environment, which we cannot know in advance. 
First make sure the PyTorch version matches your system's CUDA version [as shown on the Pytorch installation page](https://pytorch.org/get-started/locally/).
Subsequently, check the table of [matching PyTorch / torchtext versions](https://github.com/pytorch/text#installation) 
and select a matching torchtext version.

You can then install the new PyTorch / torchtext versions as follows:
```
pip install torch==1.8.1
pip install torchtext==0.9.1
```
where 1.8.1 / 0.9.1 must be replaced with a matching pair of PyTorch / torchtext versions.
