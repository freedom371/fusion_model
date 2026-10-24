import random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
import residual_model.shap.ExplainerSHAP_medical as ExplainerSHAP_medical
from torch.utils.data import DataLoader
import os
import sys
from torchvision.transforms.functional import rgb_to_grayscale

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from train.train_medical_model import MultimodalModel, MedicalDataset, create_transforms

# 设置随机种子
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


# 设备配置
device = 'cuda' if torch.cuda.is_available() else 'cpu'
sizec=128

# 定义 WrapperModel，将多输入包装成单输入
class WrapperModel(nn.Module):
    """把 (images, clinical) 拼成一个大张量，方便 SHAP 处理。"""

    def __init__(self, base, img_size=(256, 256), num_clinical_features=0):
        super().__init__()
        self.base = base
        self.img_size = img_size
        self.num_clinical_features = num_clinical_features
        self.num_image_modalities = 6
        self.num_image_channels = 3
        self.H, self.W = img_size
        # 计算各部分长度
        self.image_part_length = self.num_image_modalities * self.num_image_channels * self.H * self.W
        # self.image_part_length = 294912
        self.omics_part_length = num_clinical_features

    def forward(self, x):
        # x: [B, image_part_length + omics_part_length]
        B = x.size(0)
        # 拆分输入
        print(x.shape)
        image_flat, omics_flat = x.split([self.image_part_length, self.omics_part_length], dim=1)
        clinical_flat = omics_flat 
        # 重塑为原始格式
        images = image_flat.view(B, self.num_image_modalities, self.num_image_channels, self.H, self.W)
        clinical = clinical_flat
    
        return self.base(images, clinical)


# 加载模型和数据
def load_model(model_path, num_clinical_features, dropout=0.5):
    """加载训练好的模型"""
    model = MultimodalModel(
        num_clinical_features=num_clinical_features,
        dropout=dropout
    ).to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = checkpoint['model_state_dict'] if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint else checkpoint
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        raise RuntimeError(
            f"加载权重失败: {model_path}\n"
            f"请确认该权重与当前 MultimodalModel 结构一致（已包含 ViT 分支）。\n"
            f"原始错误: {e}"
        ) from e
    model.eval()
    return model


def resolve_model_path(model_path):
    """当配置路径不存在时，自动回退到本项目最新 checkpoint。"""
    if os.path.exists(model_path):
        return model_path

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    ckpt_dir = os.path.join(project_root, 'train', 'model_checkpoints')
    if not os.path.isdir(ckpt_dir):
        raise FileNotFoundError(f"模型路径不存在，且未找到 checkpoint 目录: {model_path}")

    candidates = [
        os.path.join(ckpt_dir, f)
        for f in os.listdir(ckpt_dir)
        if f.endswith('.pth')
    ]
    if not candidates:
        raise FileNotFoundError(f"模型路径不存在，且 checkpoint 目录为空: {model_path}")

    latest = max(candidates, key=os.path.getmtime)
    print(f"[WARN] 指定模型不存在，自动使用最新权重: {latest}")
    return latest


# 准备背景样本和测试样本
def prepare_background_and_test_samples(train_dataset, bg_size=50):
    """准备背景样本和测试样本"""
    train_loader = DataLoader(
        train_dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=False
    )
    bg_images, bg_clinical  = [], []
    test_images, test_clinical,  test_ids = [], [], []

    for i, batch in enumerate(train_loader):
        if i < bg_size:
            bg_images.append(batch['images'])
            bg_clinical.append(batch['clinical'])
        else:
            break

    bg_images = torch.cat(bg_images, dim=0).to(device)
    bg_clinical = torch.cat(bg_clinical, dim=0).to(device)
 

    return bg_images, bg_clinical



def visualize_shap(shap_values, test_x, wrap_model, test_ids, ce_mask, t2_mask, save_dir='SHAP_medical'):
    os.makedirs(save_dir, exist_ok=True)
    print("shap_values shape:", np.array(shap_values).shape)
    sv = np.array(shap_values)
    num_samples = sv.shape[0]
    print(f"Number of samples: {num_samples}")
    return_paths = []

    for sample_idx in range(num_samples):
        test_id = test_ids[sample_idx]
        if isinstance(test_id, str):
            test_id = test_id.replace('[', '').replace(']', '').replace("'", '').replace('"', '')
        else:
            test_id = str(test_id).replace('[', '').replace(']', '').replace("'", '').replace('"', '')

        print(f"Processing sample: {test_id}")
        sample_sv = sv[sample_idx]

        if sample_sv.ndim == 2:
            sample_sv = sample_sv[:, 1]

        image_sv = sample_sv[:wrap_model.image_part_length]
        omics_sv = sample_sv[wrap_model.image_part_length:]

        image_sv = image_sv.reshape(wrap_model.num_image_modalities, wrap_model.num_image_channels, wrap_model.H,
                                    wrap_model.W)

        modalities = ['CE', 'T2']
        modality_indices = [0, 3]

        for i, (modality, idx) in enumerate(zip(modalities, modality_indices)):
            plt.figure(figsize=(10, 6))
            plt.subplot(1, 2, 1)
            mask = ce_mask.to(device)
            if modality == "CE":
                mask = ce_mask.to(device)
            elif modality == "T2":
                mask = t2_mask.to(device)
            mask = mask.detach().cpu().numpy()
            print(f"mask shape: {mask.shape}")  
            mask = np.broadcast_to(mask, (3, sizec, sizec))

            modality_sv = image_sv[idx]
            print(f"modality_sv shape: {modality_sv.shape}")
            print(f"mask shape: {mask.shape}")
            # mask = mask.astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406]).reshape(3,1,1)
            std  = np.array([0.229, 0.224, 0.225]).reshape(3,1,1)

            mask = mask * std + mean
            mask = mask.mean(0)  # 变成单通道
            mask = (mask > 0.5).astype(np.float32)
            # plt.imshow(mask.transpose(1, 2, 0))
            # plt.axis('off')
            # plt.show()
            print("mask min:", mask.min())
            print("mask max:", mask.max())
            modality_sv = modality_sv * mask
            modality_sv = modality_sv.mean(0)
            vmin = max(abs(modality_sv.min()), abs(modality_sv.max()))
            im = plt.imshow(modality_sv, cmap='seismic', vmin=-vmin, vmax=vmin)
            plt.title(f'{modality} SHAP')
            plt.colorbar(im, fraction=0.046, pad=0.04)
            plt.axis('off')

            plt.subplot(1, 2, 2)
            y = np.arange(len(omics_sv))
            print(f"y : {y}")
            y=["Age","Necrosis"]
            print(f"omics_sv : {omics_sv}")
            colors = ['red' if v > 0 else 'blue' for v in omics_sv]
            plt.barh(y, omics_sv, color=colors)
            plt.xlabel("Shapley value")

            save_path = os.path.join(save_dir, f"{test_id}_{modality}_SHAP.png")
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()

            return_paths.append(save_path)
    return return_paths

# 主函数
def main():
    config = {
        'train_image_dir': r'E:\medical\train_medical_model\data\image_jpg\Test_jpg',
        'train_clinical_csv': r'E:\medical\train_medical_model\data\clinical data\Test.csv',
        'model_path': r'E:\medical\train_medical_model\train\model_checkpoints\best_model_epoch13_val_acc0.8140_seed10_20260305_163449.pth',
        'batch_size': 1,
        'bg_size': 30,
        'dropout': 0.3,
        'seed': 10
    }

    set_seed(config['seed'])

    train_transform = create_transforms(mode='train')
    train_dataset = MedicalDataset(
        image_dir=config['train_image_dir'],
        clinical_csv=config['train_clinical_csv'],
        transform=train_transform,
        mode='train'
    )

    # 获取特征数量
    num_clinical_features = len(train_dataset.clinical_df.columns) - 1  # 减去Label列
    print(f"Clinical features: {num_clinical_features}")

    # 加载模型
    config['model_path'] = resolve_model_path(config['model_path'])
    model = load_model(
        config['model_path'],
        num_clinical_features=num_clinical_features,

        dropout=config['dropout']
    )

    # 创建 WrapperModel
    wrap_model = WrapperModel(
        model,
        img_size=(128, 128),
        num_clinical_features=num_clinical_features,
    ).to(device).eval()

    # 准备背景样本和测试样本
    bg_images, bg_clinical = \
        prepare_background_and_test_samples(train_dataset, bg_size=config['bg_size'])
    # 拼接背景样本
    bg_images_flat = bg_images.view(bg_images.size(0), -1)
    bg = torch.cat([bg_images_flat, bg_clinical], dim=1)
    explainer = ExplainerSHAP_medical.GradientExplainer(wrap_model, bg)
    for example in train_dataset:
        test_images = []
        test_clinical = []
        test_radiomics = []
        test_ids = []
        test_images.append(example['images'])
        test_clinical.append(example['clinical'])

        test_id = example['id']
        test_ids.append(test_id)
        # mask 为images文件夹中ce_mask.jpg 文件
        ce_mask = example['images'][1]
        t2_mask = example['images'][4]
        # 拼接测试样本
        test_images = torch.cat(test_images, dim=0).to(device)
        test_clinical = torch.cat(test_clinical, dim=0).to(device)


        test_images_flat = test_images.view(test_images.size(0), -1)
        test_clinical = test_clinical.view(1, -1)

        test_images_flat = test_images_flat.view(1, -1)
        print(f"test_images_flat shape: {test_images_flat.shape}")
        print(f"test_clinical shape: {test_clinical.shape}")
       
        test_x = torch.cat([test_images_flat, test_clinical], dim=1)

        print(f"Background shape: {bg.shape}")
        print(f"Test shape: {test_x.shape}")

        # 计算 SHAP 值
        print("Computing SHAP values...")
        shap_values = explainer.shap_values(test_x)
        print("SHAP values computed successfully!")

        # 可视化 SHAP 值
        print(f"Visualizing SHAP for all {len(test_ids)} samples...")
        save_paths = visualize_shap(shap_values, test_x, wrap_model, test_ids, ce_mask, t2_mask)

        # 打印所有保存路径
        for save_path in save_paths:
            print(f"SHAP visualization saved to: {save_path}")


if __name__ == '__main__':
    main()
