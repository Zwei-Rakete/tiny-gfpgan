# tiny-gfpgan
Simplified GFPGAN (for inference only), strip away the original complex code, keep only the code needed for inference.

usage:
### 1. put GFPGANv1.3.pth in `models/` folder

### 2. put face images in `input_img/` folder

### 3. inference
```bash
python infer.py
```

The results are saved in `results` folder.

`arch/gfpgan_v1_clean_arch.py` -> original GFPGANv1.3 model

`arch/gfpgan_v1_clean_arch_constant.py` -> constant GFPGANv1.3 model

These two arch's outputs are highly consistent, the different is `constant arch` was modified:

1. all pytorch function coverted to pytorch class, eg: `F.conv2d()` -> `nn.Conv2d()`.

2. modified `ModulatedConv2d()` module, covert operations on weight to input, then this module can be constant.

3. other details modified.

# tiny-gfpgan

精简后的 GFPGAN 代码(只能推理，不能训练),去除原始 GFPGAN 工程复杂的代码，只保留推理所需的代码。

用法:
### 1. 将 GFPGANv1.3.pth 模型文件放到 `models/` 文件夹中

### 2. 将人脸照片放到 `input_img/` 文件夹中

### 3. 推理
```bash
python infer.py
```

修复照片保存在 `results` 文件夹中

`arch/gfpgan_v1_clean_arch.py` -> 原始 GFPGANv1.3 模型结构文件

`arch/gfpgan_v1_clean_arch_constant.py` -> 静态化的 GFPGANv1.3 模型结构文件

两个结构输出结果几乎一样，看不出差别，区别在于 `constant` 版本修改了以下几点：

1. 所有 Pytorch 函数 修改为 Pytorch 类，例如 `F.conv2d()` 修改为 `nn.Conv2d()`

2. 修改 `ModulatedConv2d()`，将原本对卷积权重 weight 的操作转换为对输入 x 的操作，因此模块从动态转换为静态

3. 其他细节修改
