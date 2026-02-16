"""DTD Dataset - Updated Structure (MVTec-like format)"""
import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import glob
import random


class DTDDataset(Dataset):
    def __init__(self, root, train=True, category=None, fewshot=0, transform=None, gt_target_transform=None):
        super(DTDDataset, self).__init__()
        
        # DTD数据集的新类别结构（基于实际目录）
        self.categories = [
            'Blotchy_099', 'Fibrous_183', 'Marbled_078', 'Matted_069', 
            'Mesh_114', 'Perforated_037', 'Stratified_154', 
            'Woven_001', 'Woven_068', 'Woven_104', 'Woven_125', 'Woven_127'
        ]
        
        self.train = train  # 布尔值：True=训练集，False=测试集
        self.category = category  # 指定类别（None表示所有类别）
        self.fewshot = fewshot  # 少样本数量（仅训练时生效）
        self.root = os.path.join(root, 'DTD')  # 数据集根目录
        self.transform = transform  # 图像预处理
        self.gt_target_transform = gt_target_transform  # 掩码预处理
        self.preprocess()  # 预处理：收集所有图像/掩码路径和标签
        self.update(self.category)  # 更新当前类别对应的样本
        # 校验数据长度一致性
        assert len(self.cur_img_paths) == len(self.cur_img_labels)
        assert len(self.cur_img_paths) == len(self.cur_img_categories)
        assert len(self.cur_img_paths) == len(self.cur_gt_paths)
        self.dataset_name = 'dtd'  # 数据集名称标识
        
    def preprocess(self):
        """预处理：遍历所有类别和阶段，收集图像路径、掩码路径和标签"""
        # 初始化存储结构：phase分为'train'和'test'
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
                cat_root = os.path.join(self.root, category)
                
                # 训练集：仅包含train/good（正常样本）
                if phase == 'train':
                    img_dir = os.path.join(cat_root, 'train', 'good')
                    if os.path.exists(img_dir):
                        # 收集所有正常样本图像路径
                        img_paths = glob.glob(os.path.join(img_dir, "*.png")) + glob.glob(os.path.join(img_dir, "*.jpg"))
                        img_paths.sort()  # 排序确保一致性
                        self.img_paths['train'][category].extend(img_paths)
                        self.gt_paths['train'][category].extend([None] * len(img_paths))  # 训练集无掩码
                        self.labels['train'][category].extend([0] * len(img_paths))  # 0表示正常
                
                # 测试集：包含test/good（正常）和test/bad（异常）
                elif phase == 'test':
                    test_dir = os.path.join(cat_root, 'test')
                    if not os.path.exists(test_dir):
                        continue  # 跳过不存在测试集的类别
                    
                    # 遍历test下的子目录（good=正常，bad=异常）
                    defect_types = os.listdir(test_dir)
                    for defect_type in defect_types:
                        # 跳过掩码目录（单独处理）
                        if defect_type == 'ground_truth':
                            continue
                        
                        defect_dir = os.path.join(test_dir, defect_type)
                        if not os.path.isdir(defect_dir):
                            continue  # 跳过非目录文件
                        
                        # 收集当前缺陷类型的图像路径
                        img_paths = glob.glob(os.path.join(defect_dir, "*.png")) + glob.glob(os.path.join(defect_dir, "*.jpg"))
                        img_paths.sort()
                        
                        # 标签：good=0（正常），bad=1（异常）
                        labels = [0 if defect_type == 'good' else 1 for _ in img_paths]
                        self.labels['test'][category].extend(labels)
                        self.img_paths['test'][category].extend(img_paths)
                        
                        # 处理掩码路径（仅异常样本需要）
                        if defect_type != 'good':
                            # 掩码路径：ground_truth/bad（在类别根目录下，不在test目录下）
                            gt_dir = os.path.join(cat_root, 'ground_truth', 'bad')
                            if os.path.exists(gt_dir):
                                # 获取对应异常样本的掩码文件
                                gt_paths = []
                                for img_path in img_paths:
                                    img_filename = os.path.basename(img_path)
                                    name, ext = os.path.splitext(img_filename)
                                    # 构造对应的掩码文件名：原文件名 + _mask + 扩展名
                                    gt_filename = f"{name}_mask{ext}"
                                    gt_path = os.path.join(gt_dir, gt_filename)
                                    if os.path.exists(gt_path):
                                        gt_paths.append(gt_path)
                                    else:
                                        gt_paths.append(None)  # 如果找不到对应掩码
                                self.gt_paths['test'][category].extend(gt_paths)
                            else:
                                self.gt_paths['test'][category].extend([None] * len(img_paths))
                        else:
                            # 正常样本无掩码
                            self.gt_paths['test'][category].extend([None] * len(img_paths))
    
    def update(self, category=None):
        """根据指定类别更新当前样本列表（训练/测试阶段由self.train决定）"""
        self.category = category
        self.cur_img_paths, self.cur_gt_paths, self.cur_img_labels, self.cur_img_categories = [], [], [], []
        
        # 确定当前阶段（train或test）
        phase = 'train' if self.train else 'test'
        
        # 收集指定类别或所有类别的样本
        if self.category is not None:
            # 仅加载指定类别的样本
            self.cur_img_paths = self.img_paths[phase][self.category]
            self.cur_gt_paths = self.gt_paths[phase][self.category]
            self.cur_img_labels = self.labels[phase][self.category]
            self.cur_img_categories = [self.category] * len(self.cur_img_paths)
        else:
            # 加载所有类别的样本
            for cat in self.categories:
                self.cur_img_paths.extend(self.img_paths[phase][cat])
                self.cur_gt_paths.extend(self.gt_paths[phase][cat])
                self.cur_img_labels.extend(self.labels[phase][cat])
                self.cur_img_categories.extend([cat] * len(self.img_paths[phase][cat]))
        
        # 少样本处理（仅训练时生效）
        if self.train and self.fewshot != 0:
            # 随机选择fewshot个样本
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
        if self.cur_gt_paths[idx] is not None and os.path.exists(self.cur_gt_paths[idx]):
            gt = np.array(Image.open(self.cur_gt_paths[idx]))
        else:
            # 若无掩码，生成与图像同尺寸的全0数组
            gt = np.zeros((img.size[1], img.size[0]), dtype=np.uint8)
        gt = Image.fromarray(gt)
        
        # 应用预处理
        if self.transform is not None:
            img = self.transform(img)
        if self.gt_target_transform is not None:
            gt = self.gt_target_transform(gt)
        
        return img, label, gt, category, img_path
