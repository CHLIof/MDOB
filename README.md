
# Multi-Dataset Object Detection .



This project is based on [mmdetection](https://github.com/open-mmlab/mmdetection)



### Introduction



### Datasets Preparation

Fisrst of all, download the datasets: [NWPU](https://www.sciencedirect.com/science/article/pii/S0924271614002524?via%3Dihub), 
[RSOD](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7827088), [UCAS](http://lamp.ucas.ac.cn/downloads/publication/ICIP2015_ZhuHaigang.pdf).

Then, divide them respectively into training and testing sets in an 8:2 ratio, and annotate files in COCO format
(We also provide annotated files in coco format that have already been divided at [datasets](datasets)).



<a id="section1"></a>
### Teacher Model

You can download the [teacher model](https://pan.baidu.com/s/1PyG4ZC3D53D9kprkebgeOg) (extracted code:dahb) which is a GLIP model finetuned on the three datasets(NWPU, RSOD, UCAS): 



### Compilation

You can install the environment according to mmdetection's [installation instructions](https://mmdetection.readthedocs.io/en/latest/get_started.html) or follow belows:

Create a conda environment and activate it:

```
conda create --name openmmlab python=3.8 -y
conda activate openmmlab
```

Install PyTorch:

```
conda install pytorch torchvision -c pytorch
```

Install MMEngine and MMCV using MIM:

```
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
```


Install MMDetection:


```
# Enter the project directory
pip install -v -e .
# "-v" means verbose, or more output
# "-e" means installing a project in editable mode,
# thus any local modifications made to the code will take effect without reinstallation.
```


### Train

Before training, modify the [configuration file](configs/MDOB/MDOB.py) according to your path.

First, modify the path of pretrained bockbone model (or let it download automatically).

second, modify the path of teacher model, which you can download at [Teacher Model](#section1).

Third, modify the path of datasets and their annotation files.

Forth, modify the save path.


Then,  simply run:

```
python tools/train.py configs/MDOB/MDOB.py
```



### Test

To evaluate the detection performance of each datasets, simply run:

```
python tools/test.py configs/MDOB/MDOB.py /path/final.path  
```


Results of our model for 3 datasets:

| Method | Backbone | NWPU | RSOD | UCAS | download |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| Ours | Swin-tiny | 61.4 | 67.4 | 66.4 | [model](https://pan.baidu.com/s/1TeQvESJ0-z-wXxJ0aHYtLg) (extracted code:bjp6)|






