# NTIRE2025_DGL_DeRainDrop

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
