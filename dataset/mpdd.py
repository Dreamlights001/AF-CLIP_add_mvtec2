"""MPDD Dataset"""
import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import glob
import random


class MPDDDataset(Dataset):
    def __init__(self, root, train=True, category=None, fewshot=0, transform=None, gt_target_transform=None):
        super(MPDDDataset, self).__init__()
        # MPDD 数据集的所有类别
        self.categories = [
            'bracket_black', 'bracket_brown', 'bracket_white',
            'connector', 'metal_plate', 'tubes'
        ]
        
        self.train = train  # 布尔值：True=训练集，False=测试集
        self.category = category  # 指定类别（None 表示所有类别）
        self.fewshot = fewshot  # 少样本数量（仅训练时生效）
        self.root = os.path.join(root, 'mpdd')  # 数据集根目录
        self.transform = transform  # 图像预处理
        self.gt_target_transform = gt_target_transform  # 掩码预处理
        self.preprocess()  # 预处理：收集所有图像/掩码路径和标签
        self.update(self.category)  # 更新当前类别对应的样本
        # 校验数据长度一致性
        assert len(self.cur_img_paths) == len(self.cur_img_labels)
        assert len(self.cur_img_paths) == len(self.cur_img_categories)
        assert len(self.cur_img_paths) == len(self.cur_gt_paths)
        self.dataset_name = "mpdd"  # 数据集名称标识
        
    def preprocess(self):
        """预处理：遍历所有类别和阶段，收集图像路径、掩码路径和标签"""
        # 初始化存储结构：phase 分为 'train' 和 'test'
        self.img_paths = {'train': {cat: [] for cat in self.categories}, 
                          'test': {cat: [] for cat in self.categories}}
        self.gt_paths = {'train': {cat: [] for cat in self.categories}, 
                         'test': {cat: [] for cat in self.categories}}
        self.labels = {'train': {cat: [] for cat in self.categories}, 
                       'test': {cat: [] for cat in self.categories}}
        
        # 遍历训练集和测试集
        for phase in ['train', 'test']:
            for category in self.categories:
                # 拼接当前类别的根目录
                img_dir = os.path.join(self.root, category)
                if not os.path.exists(os.path.join(img_dir, phase)):
                    continue
                
                # 遍历当前阶段下的所有缺陷类型
                defect_types = os.listdir(os.path.join(img_dir, phase))
                for defect_type in defect_types:
                    # 处理正常样本
                    if defect_type == 'good':
                        img_paths = glob.glob(os.path.join(img_dir, phase, defect_type) + "/*.png")
                        img_paths.sort()
                        self.img_paths[phase][category].extend(img_paths)
                        self.gt_paths[phase][category].extend([None] * len(img_paths))
                        self.labels[phase][category].extend([0] * len(img_paths))
                    # 处理异常样本
                    else:
                        img_paths = glob.glob(os.path.join(img_dir, phase, defect_type) + "/*.png")
                        gt_paths = glob.glob(os.path.join(img_dir, 'ground_truth', defect_type) + "/*.png")
                        img_paths.sort()
                        gt_paths.sort()
                        self.img_paths[phase][category].extend(img_paths)
                        self.gt_paths[phase][category].extend(gt_paths)
                        self.labels[phase][category].extend([1] * len(img_paths))
    
    def update(self, category=None):
        """根据指定类别更新当前样本列表（训练/测试阶段由 self.train 决定）"""
        self.category = category
        self.cur_img_paths, self.cur_gt_paths, self.cur_img_labels, self.cur_img_categories = [], [], [], []
        
        # 确定当前阶段（train 或 test）
        if self.train:
            phase = 'train'
        else:
            phase = 'test'
        
        # 收集指定类别或所有类别的样本
        if self.category is not None:
            # 仅加载指定类别的样本
            self.cur_img_paths = self.img_paths[phase][self.category]
            self.cur_gt_paths = self.gt_paths[phase][self.category]
            self.cur_img_labels = self.labels[phase][self.category]
            self.cur_img_categories = [self.category] * len(self.cur_img_paths)
        else:
            # 加载所有类别的样本
            for category in self.categories:
                self.cur_img_paths.extend(self.img_paths[phase][category])
                self.cur_gt_paths.extend(self.gt_paths[phase][category])
                self.cur_img_labels.extend(self.labels[phase][category])
                self.cur_img_categories.extend([category] * len(self.img_paths[phase][category]))
    
        # 少样本处理（仅训练时生效）
        if self.train and self.fewshot != 0:
            # 随机选择 fewshot 个样本
            randidx = np.random.choice(len(self.cur_img_paths), size=self.fewshot, replace=False)
            self.cur_img_paths = [self.cur_img_paths[idx] for idx in randidx]
            self.cur_gt_paths = [self.cur_gt_paths[idx] for idx in randidx]
            self.cur_img_labels = [self.cur_img_labels[idx] for idx in randidx]
            self.cur_img_categories = [self.cur_img_categories[idx] for idx in randidx]
    
    def __len__(self):
        """返回当前样本数量"""
        return len(self.cur_img_paths)

    def __getitem__(self, idx):
        """加载单个样本（图像、标签、掩码、类别、路径）"""
        category = self.cur_img_categories[idx]
        img_path = self.cur_img_paths[idx]
        label = self.cur_img_labels[idx]
        
        # 加载图像（RGB格式）
        img = Image.open(img_path).convert('RGB')
        
        # 加载掩码（灰度图，正常样本为全0）
        if self.cur_gt_paths[idx] is not None:
            gt = np.array(Image.open(self.cur_gt_paths[idx]))
        else:
            gt = np.zeros((img.size[1], img.size[0]), dtype=np.uint8)
        gt = Image.fromarray(gt)
        
        # 应用预处理
        if self.transform is not None:
            img = self.transform(img)
        if self.gt_target_transform is not None:
            gt = self.gt_target_transform(gt)
        
        return img, label, gt, category, img_path
