### Introduction
This is the data repository for CBPH-Net, and further usage details will be provided after the acceptance of the paper. The dataset STBD-08, created in this study, is available for download. However, due to privacy concerns related to the research topic, please email the corresponding author of the paper to obtain a copy of the agreement. This agreement ensures that the dataset will not be used for unintended academic purposes.

### Please do not send me email any more, I will graduate this year and the edu. email will withdraw. 
I am sorry for the prevoious error code, it has some problems, reasons and solution you can see issue #2.
After you download the code，please delete model/CBPH.py, it has been rewritten into model/common.py, but you can still read it to learn. 
I update a new version, you can just run it by "python train.py --cfg models/CBPH-Net.yaml --data data/voc_bm.yaml --batch-size 64 --epoch 100". 

### Please do not send me email any more, I will graduate this year and the edu. email will withdraw. 
For the dataset, please consult "https://github.com/Whiffe/SCB-dataset", it has more related datasets and i have sent my dataset to this researcher.

### 请不要再发送邮件进行咨询，我将于2024.5月从华中师大毕业，教育邮箱将会被收回，未来也无意在这个领域深耕，请勿打扰，敬请谅解。
之前的代码有一些小错误，详情和解决请见我在issues #2里的回答。我对于之前存在的错误十分抱歉。
我上传了新的代码，并确保其可以运行。重新注明了如何安装CBPH头，如果您不想使用，注释掉代码中对应的部分，并在yaml文件中将CBPH更换为Detect即可。
![image](https://github.com/icedle/CBPH-Net/assets/52993977/b1472e76-991d-45c7-af7d-d73f8166c0a1)
如果您想关闭其中的模块，请自行注释掉不需要的部分就好。代码仓库是基于YOLOv5的，网上有很多教程，我的代码基本上也是在其基础上进行修改的（在这里很抱歉地回复某个咨询者，我没能力自行创作全新的模型，只能在这里修修改改，如果有我就会去发顶会而不是2区不知名期刊）。

### 请不要再发送邮件进行咨询，我将于2024.5月从华中师大毕业，教育邮箱将会被收回，未来也无意在这个领域深耕，请勿打扰，敬请谅解。
关于数据集，请移步https://github.com/Whiffe/SCB-dataset，这里有更丰富的数据集资源，之前的数据集我也授权给这位老师了。
今后不会再回复咨询邮件，因为我将于2024/5月从华中师范大学毕业，教育邮箱也会被收回，未来也无意在这个领域进行深耕，请勿打扰。

### 现行代码可以100%运行，如果您想要调整模型性能，可以将CBPH-Net.yaml中将通道大小改为0.5和0.5或者1.0 和 1.0（请参照YOLOv5模型）

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
