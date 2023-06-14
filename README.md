### Introduction

This is the data repository for CBPH-Net, and further usage details will be provided after the acceptance of the paper. The dataset STBD-08, created in this study, is available for download. However, due to privacy concerns related to the research topic, please email the corresponding author of the paper to obtain a copy of the agreement. This agreement ensures that the dataset will not be used for unintended academic purposes.

### Install

```
github clone https://github.com/icedle/CBPH-Net`

cd CBPH-Net

pip install -r requirements
```



#### How to install CBPH

The default detection head used is the one from YOLOv5. However, we have integrated CBPH into the project (refer to models/CBPH.py). If you want to enable CBPH as the detection head, please follow the steps below:

```
1. Go to models/yolo.py
Find code in line 184

def _profile_one_layer(self, m, x, dt):
	c = isinstance(m, Detect): 

and modify it as:
	
	c = isinstance(m, Detect, CBPH):

2. Go to models/yolo.py
Find code in line 184
	if isinstance(m, Detect):

and modify it as:
	
	c = isinstance(m, Detect, CBPH):
3. Go to models/CBPH-Net.yaml
Replace the "Detect" in the last line with "CBPH".
```

### Train

```
We give an example, more details please see train.py
python trian.py --cfg models/CBPH-Net.yaml --batch-size 32 --epoch 100 
```

### Inference

```
Please see val.py
```

We give an example

![results](https://github.com/icedle/CBPH-Net/blob/main/imgs/results.png)

### Dataset

The dataset format is VOC (.xml). You can use to convert it to COCO(.json) or TXT format. We already provide a script to transfer  xml format to txt format (Please see prepare_data.py).

![datasets](https://github.com/icedle/CBPH-Net/blob/main/imgs/datasets.png)

### Performance