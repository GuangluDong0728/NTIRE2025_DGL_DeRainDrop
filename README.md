# NTIRE 2025 The First Challenge on Day and Night Raindrop Removal for Dual-Focused Images: GSA2Step by DGL_DeRainDrop 
## Team Members: </br>
[Guanglu Dong](https://github.com/GuangluDong0728)\*,
[Xin Lin](https://linxin0.github.io/),
Siyuan Liu,
Tianheng Zheng,
Jiayu Zhong,
Shouyi Wang,
Xiangtai Li,
Lu Qi,
Chao Ren

## Introduction:
Our code is implemented based on [BasicSR](https://github.com/XPixelGroup/BasicSR). The network structure of our designed GSA2Step is in "class NAFNet_CLIP_2Stage(nn.Module)" of `basicsr/archs/NAFNet_arch.py`, and the test configuration file is in `options/test/Derain/test.yml`. Please place the dataset in the `datasets` folder and modify the configuration file to test our model according to the following process.

## Our Network Structure
Our GSA2Step framework is detailed in figure below, and it mainly includes 3 modules: 2 NAFNet (a 32 width version and a 64 width version), a global semantic attention (GSA) module, and a CLIP~\cite{CLIP} image encoder.
![alt text](assert/F1.png)

# Environment Prepare
You can refer to the environment preparation process of [BasicSR](https://github.com/XPixelGroup/BasicSR), which mainly includes the following two steps:

1. 

    ```bash
    pip install -r requirements.txt
    ```

2. 

    ```bash
    python setup.py develop
    ```

# Downloading Our Weights

1. **Download Pretrained Weights:**
   - Navigate to [this link](https://drive.google.com/drive/folders/1y2kA1qQcLBHChQKlFzyvIQOmyV0bUq-J?usp=sharing) to access the our weights. You only need to download the "net_g_final.pth" file.
   
2. **Save to `experiments` Directory:**
   - Once downloaded, place the weights into the `experiments` directory.
  
# Validation and Testing

## modify the config file
To validate the our model, you need to modify the paths in the configuration file. Open the `options/test/Derain/test.yml` file and update the paths, and just run the command:

```bash
python basicsr/test.py -opt options/test/Derain/test.yml
```
# Our Results
Our results on validation dataset and test dataset can be download at [this link](https://drive.google.com/drive/folders/15MCuydmLbWZ3EhQ5Tjp5G9UveGCuAKcw?usp=sharing).

# Acknowledgements

This project is built on source codes shared by [BasicSR](https://github.com/XPixelGroup/BasicSR), [NAFNet](https://github.com/megvii-research/NAFNet), and [pyiqa](https://github.com/chaofengc/IQA-PyTorch).
