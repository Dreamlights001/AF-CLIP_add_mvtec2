"""dataset"""
import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import glob
from torchvision import transforms

class BrainMRIDataset(Dataset):
    def __init__(self, root, train=True, category=None, transform=None, gt_target_transform=None):
        super(BrainMRIDataset, self).__init__()
        self.img_root = os.path.join(root, 'BrainMRI')
        self.train = train
        self.transform = transform
        self.gt_target_transform = gt_target_transform  # 保存掩码变换
        self.preprocess()
        self.categories = ['brainmri']
        self.dataset_name = "brainmri"
        self.category = "brainmri"
        
        
    def preprocess(self):
        normal_img_paths, anomaly_img_paths = [], []
        for t in  ['jpeg', 'jpg', 'JPG', 'JEPG', 'png', 'PNG']:
            normal_img_paths.extend(glob.glob(os.path.join(self.img_root, 'no') + "/*." + str(t)))
            anomaly_img_paths.extend(glob.glob(os.path.join(self.img_root, 'yes') + "/*." + str(t)))
        anomaly_img_paths.sort()
        normal_img_paths.sort()
        self.img_paths = normal_img_paths + anomaly_img_paths
        self.labels = np.concatenate([np.zeros(len(normal_img_paths)), np.ones(len(anomaly_img_paths))])
    
    def update(self, category):
        pass
             
    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        category = "brainmri"
        img_path = self.img_paths[idx]
        label = self.labels[idx]
        
        # BrainMRI数据集没有像素级标注，仅为分类任务设计
        # 生成与图像尺寸相同的全0掩码（用于分类任务的占位符）
        img = Image.open(img_path).convert('RGB')
        gt = np.zeros((img.size[1], img.size[0]), dtype=np.uint8)  # 全0掩码作为占位符
        
        gt = Image.fromarray(gt)
        
        if self.transform is not None:
            img = self.transform(img)
        
        # 应用掩码变换（重要：确保掩码能被正确批处理）
        if self.gt_target_transform is not None:
            gt = self.gt_target_transform(gt)
        
        return img, label, gt, category, img_path
    