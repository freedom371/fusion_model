# LazySlide + UNI2 Demo

这个 demo 用 `LazySlide + UNI2` 对病理大图像进行基础模型特征提取，流程是：

`WSI -> tissue detection -> tiling -> UNI2 tile features -> mean slide feature`

输出包括：

- tile 级特征：`.npy`
- slide 级聚合特征：`.npy`
- tile 坐标元数据：`.csv`
- 运行摘要：`summary.json`
- 可选的 LazySlide backed store：`.zarr`

## 1. 安装

```powershell
python -m pip install -r requirements.txt
```

## 2. UNI2 权限

`UNI2` 是 Hugging Face gated model。第一次使用前需要：

1. 申请访问 `MahmoodLab/UNI2-h`
2. 准备一个 Hugging Face token

你可以用环境变量：

```powershell
$env:HF_TOKEN="your_hf_token"
```

或者直接在命令行里传：

```powershell
python .\demo_lazyslide_uni2.py --wsi C:\path\to\slide.svs --hf-token your_hf_token
```

如果你已经把权重下载到本地，也可以使用：

```powershell
python .\demo_lazyslide_uni2.py --wsi C:\path\to\slide.svs --model-path C:\path\to\pytorch_model.bin
```

## 3. 直接改代码里的配置

脚本顶部有一个 `CONFIG`：

```python
CONFIG = {
    "use_sample": True,
    "wsi_path": None,
    "output_dir": Path("outputs") / "lazyslide_uni2_demo",
    "model": "uni2",
    ...
}
```

推荐这样用：

- 跑官方样例：把 `use_sample=True`
- 跑自己的切片：把 `use_sample=False`，然后填写 `wsi_path`
- 切换基础模型：改 `model`，比如 `uni2`、`phikon`、`virchow2`
- 有 UNI2 权限：填写 `hf_token` 或设置环境变量 `HF_TOKEN`

改完以后直接运行：

```powershell
python .\demo_lazyslide_uni2.py
```

## 4. 跑内置样例

先用 LazySlide 自带的小样本切片做冒烟测试：

```powershell
python .\demo_lazyslide_uni2.py
```

如果没有 `UNI2` 权限，也可以先把模型换成公开模型验证流程：

```powershell
# 在 CONFIG 里改成:
# "use_sample": True
# "model": "phikon"
```

## 5. 跑你们自己的 WSI

把 `CONFIG` 改成类似这样：

```python
CONFIG = {
    "use_sample": False,
    "wsi_path": Path(r"C:\data\case_001.svs"),
    "output_dir": Path("outputs") / "case_001_uni2",
    "model": "uni2",
    "hf_token": "your_hf_token",
    "batch_size": 8,
    "num_workers": 0,
    "save_zarr": True,
    ...
}
```

常用参数：

- `tile_size=224`：tile 大小
- `stride=224`：stride，默认等于 tile size
- `mpp=0.5`：切 tile 的物理分辨率
- `device="cuda"`：显卡推理
- `model="uni2"`：默认就是 `uni2`
- `list_models=True`：列出 LazySlide 当前内置模型

## 6. 输出说明

默认输出目录下会生成：

- `*_tile_features.npy`：每个 tile 一行特征
- `*_slide_features.npy`：全切片聚合特征
- `tile_metadata.csv`：tile 索引和坐标
- `summary.json`：本次运行参数和形状信息
- `thumbnail.png`：若可用则导出缩略图

## 7. 备注

- 这个 demo 目前默认做 `mean` 聚合，适合先拿到一个 slide-level representation。
- 如果后面你们想扩展成 `UNI2/Virchow/Phikon/GigaPath` 多模型批量抽特征，我可以继续把它改成一个批处理版本。
