### Introduction
This is the data repository for CBPH-Net, and further usage details will be provided after the acceptance of the paper. The dataset STBD-08, created in this study, is available for download. However, due to privacy concerns related to the research topic, please email the corresponding author of the paper to obtain a copy of the agreement. This agreement ensures that the dataset will not be used for unintended academic purposes.


I am sorry for the prevoious error code, it has some problems, reasons and solution you can see issue #2.
After you download the code，please delete model/CBPH.py, it has been rewritten into model/common.py, but you can still read it to learn. 
I update a new version, you can just run it by "python train.py --cfg models/CBPH-Net.yaml --data data/voc_bm.yaml --batch-size 64 --epoch 100". 


For the dataset, please consult "https://github.com/Whiffe/SCB-dataset", it has more related datasets and i have sent my dataset to this researcher.


### Install

```
github clone https://github.com/icedle/CBPH-Net`

cd CBPH-Net

pip install -r requirements
```

#### How to install CBPH

The default detection head used is the one from YOLOv5. However, we have integrated CBPH into the project (refer to models/CBPH.py). If you want to enable CBPH as the detection head, please follow the steps below:

```
1. Go to yolo.py, find isinstance (m, detect), and add the following code after it:
 if isinstance(m, CBPH):
    s = 256  # 2x min stride
    m.inplace = self.inplace
    m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))])  # forward
    m.anchors /= m.stride.view(-1, 1, 1)
    check_anchor_order(m)
    self.stride = m.stride


2. Go to models/yolo.py, find:

def _profile_one_layer(self, m, x, dt):
	c = isinstance(m, Detect): 

and modify it as:
	
	c = isinstance(m, Detect, CBPH):

3. Go to models/yolo.py, find:

def _apply(self, fn)
	if isinstance(m, Detect):

and modify it as:
	
	c = isinstance(m, Detect) or isinstance (m, CBPH):

4. Go to models/yolo.py, find "parse_model" function, and add the following code: 
    elif m is CBPH:
        args.append([ch[x] for x in f])
        if isinstance(args[1], int):  # number of anchors
            args[1] = [list(range(args[1] * 2))] * len(f)


5. Go to models/CBPH-Net.yaml
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

![results](https://github.com/icedle/CBPH-Net/blob/main/imgs/results.jpg)

### Dataset

The dataset format is VOC (.xml). You can use to convert it to COCO(.json) or TXT format. We already provide a script to transfer  xml format to txt format (Please see prepare_data.py).

![datasets](https://github.com/icedle/CBPH-Net/blob/main/imgs/datasets.png)

### Performance
