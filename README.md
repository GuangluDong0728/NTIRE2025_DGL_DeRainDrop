# NTIRE 2025 The First Challenge on Day and Night Raindrop Removal for Dual-Focused Images: GSA2Step by DGL_DeRainDrop 
## Team Members: </br>
[Guanglu Dong](https://github.com/GuangluDong0728)\*,
[Xin Lin](https://linxin0.github.io/)\*,
Siyuan Liu,
Tianheng Zheng,
Jiayu Zhong,
Shouyi Wang,
Xiangtai Li,
Lu Qi,
Chao Ren

# Environment prepare

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
   - Navigate to [this link](https://drive.google.com/drive/folders/1Qfz8cbB9jHcTzAAQEpPn7gvSkuiNnovN?usp=sharing) to access the our weights.
   
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
