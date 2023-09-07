import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset


class DriveDataset(Dataset):
    # 这里定义的是DRIVE数据集的读取
    def __init__(self, root: str, train: bool, transforms=None):  # 这里的root是指的数据集的根目录，train是指的是训练集还是测试集
        super(DriveDataset, self).__init__()  # 这里调用父类的初始化函数
        self.flag = "training" if train else "test"  # 这里根据train的值来确定是训练集还是测试集
        data_root = os.path.join(root, "DRIVE", self.flag)  # 这里的data_root是指的数据集的根目录
        assert os.path.exists(data_root), f"path '{data_root}' does not exists."
        self.transforms = transforms  # 这里的transforms是进行数据增强的操作
        img_names = [i for i in os.listdir(os.path.join(data_root, "images")) if i.endswith(".tif")]
        self.img_list = [os.path.join(data_root, "images", i) for i in img_names]  # 这里的img_list是获取图片的路径
        self.manual = [os.path.join(data_root, "1st_manual", i.split("_")[0] + "_manual1.gif")
                       for i in img_names]  # 这是获取mask的路径  这是人工标注的mask，也就是ground truth
        # 这里的可以解释一下，因为DRIVE数据集是眼球数据集，这里self.manual是获取的眼球的血管的mask，且仅有血管和背景，而self.roi_mask是获取的眼球的mask，且仅有眼球，后面的再将这两者融合
        # check files 这里是检查文件是否存在
        for i in self.manual:
            if os.path.exists(i) is False:
                raise FileNotFoundError(f"file {i} does not exists.")

        self.roi_mask = [os.path.join(data_root, "mask", i.split("_")[0] + f"_{self.flag}_mask.gif")
                         for i in img_names]  # 这里是获取roi_mask的路径
        # check files  
        for i in self.roi_mask:
            if os.path.exists(i) is False:
                raise FileNotFoundError(f"file {i} does not exists.")

    def __getitem__(self, idx):
        img = Image.open(self.img_list[idx]).convert('RGB')  # 这里是将图片转换成RGB格式
        manual = Image.open(self.manual[idx]).convert('L')  # 这里是将mask转换成灰度图
        manual = np.array(manual) / 255  # 这里是将mask进行归一化，因为黑色的部分是0，白色的部分是255，所以这里除以255，使得mask的值为1，让网络更好的学习
        roi_mask = Image.open(self.roi_mask[idx]).convert('L') # 这里是将roi_mask转换成灰度图
        roi_mask = 255 - np.array(roi_mask)  # 这里是将roi_mask进行反转，因为roi_mask中，眼球的部分是255，背景是0，所以这里进行反转，使得眼球的部分是0，背景是255
        mask = np.clip(manual + roi_mask, a_min=0, a_max=255) # 这里是将mask和roi_mask进行融合

        # 这里转回PIL的原因是，transforms中是对PIL数据进行处理
        mask = Image.fromarray(mask)  # 这里是将mask转换成PIL格式

        if self.transforms is not None:
            img, mask = self.transforms(img, mask)  # 这里是对图片和mask进行数据增强的操作

        return img, mask  # 这里返回的是经过处理后的图片和mask

    def __len__(self):
        return len(self.img_list)  #    这里返回的是图片的数量

    @staticmethod
    def collate_fn(batch): # 这是对batch进行处理的函数
        images, targets = list(zip(*batch))  # 这里是将batch中的图片和mask分开
        batched_imgs = cat_list(images, fill_value=0)  # 这里是将图片进行拼接，使得图片的大小一样，并且填充的值是0
        batched_targets = cat_list(targets, fill_value=255)  # 这里是将mask进行拼接，使得mask的大小一样，并且填充的值是255
        return batched_imgs, batched_targets


def cat_list(images, fill_value=0):  
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))  # 这里是获取图片中最大的尺寸
    batch_shape = (len(images),) + max_size  # 这里是获取batch的大小，方法就是图片的数量（20）加上图片的最大尺寸
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)  # 这里是创建一个batch的tensor，大小是batch_shape，并且填充的值是0
    for img, pad_img in zip(images, batched_imgs):  
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)  # 通过切片将图片的数据填充到batched_imgs中
    return batched_imgs

