import os
import time
import datetime

import torch
# 从src文件夹中导入UNet
from src import UNet
#导入train_utils.py中的函数
from train_utils import train_one_epoch, evaluate, create_lr_scheduler
#导入我们的数据集
from my_dataset import DriveDataset
import transforms as T


class SegmentationPresetTrain:
    # 这里定义的是训练时的数据增强，包括随机裁剪、随机水平翻转、随机垂直翻转，以及归一化平均值和标准差
    def __init__(self, base_size, crop_size, hflip_prob=0.5, vflip_prob=0.5,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        min_size = int(0.5 * base_size)
        max_size = int(1.2 * base_size)  # 这里的base_size是指的输入图片的大小

        trans = [T.RandomResize(min_size, max_size)] # 随机裁剪，裁剪的大小在[0.5*base_size, 1.2*base_size]之间
        if hflip_prob > 0:
            trans.append(T.RandomHorizontalFlip(hflip_prob))  # 随机水平翻转
        if vflip_prob > 0:
            trans.append(T.RandomVerticalFlip(vflip_prob)) # 随机垂直翻转
        trans.extend([  # 进行随机裁剪，裁剪的大小是crop_size，转变为tensor，然后进行归一化
            T.RandomCrop(crop_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]) 
        self.transforms = T.Compose(trans)  # 这里使用的是Compose，将上面定义的操作组合起来

    def __call__(self, img, target):
        return self.transforms(img, target)  # 这里返回的是经过处理后的图片和mask


class SegmentationPresetEval:  
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.transforms = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])  # 将图片转变成张量，再进行归一化

    def __call__(self, img, target):
        return self.transforms(img, target)


def get_transform(train, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    base_size = 565
    crop_size = 480  # 设定裁剪尺寸

    if train:
        return SegmentationPresetTrain(base_size, crop_size, mean=mean, std=std)
    else:
        return SegmentationPresetEval(mean=mean, std=std)  #根据训练和验证来设定图像数据增强


def create_model(num_classes):
    #  这个函数，创建我们的UNet模型，设定我们的输入通道为3，识别类别为就2个就黑白两个，base_c就是最开始的通道数，这是设定是32通道
    model = UNet(in_channels=3, num_classes=num_classes, base_c=32)  
    return model


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu") # 对传进去的设备进行指定
    batch_size = args.batch_size  # 根据我们的参数来设定batch_size
    # segmentation nun_classes + background
    num_classes = args.num_classes + 1

    # using compute_mean_std.py
    mean = (0.709, 0.381, 0.224)
    std = (0.127, 0.079, 0.043)

    # 用来保存训练以及验证过程中信息
    results_file = "results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    train_dataset = DriveDataset(args.data_path,  # 设定训练的DriveDataset，传进去数据集的路径，设定为训练，再传进数据增广的参数
                                 train=True,
                                 transforms=get_transform(train=True, mean=mean, std=std))

    val_dataset = DriveDataset(args.data_path,  # 这是验证的数据集创建，再传进去参数
                               train=False,
                               transforms=get_transform(train=False, mean=mean, std=std))

    num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # 进程数是根据cpu的进程数，8，batch_size这几个来取最小值
    # 创建训练的dataloader,再传进去训练的数据集，batch_size的大小，num_workers，进行shuffle打乱，把数据放入GPU中
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               num_workers=num_workers,
                                               shuffle=True,
                                               pin_memory=True,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,  # 同上
                                             batch_size=1,
                                             num_workers=num_workers,
                                             pin_memory=True,
                                             collate_fn=val_dataset.collate_fn)

    model = create_model(num_classes=num_classes)
    model.to(device)

    params_to_optimize = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.SGD(
        params_to_optimize,
        lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
    )  # 对优化器进行设定，这里使用的是SGD，传入参数

    # 对混合精度进行设定，如果使用混合精度，就使用GradScaler，否则就为None
    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    # 创建学习率更新策略，这里是每个step更新一次(不是每个epoch)
    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs, warmup=True)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')  # 传入路径
        model.load_state_dict(checkpoint['model'])  # 加载模型
        optimizer.load_state_dict(checkpoint['optimizer'])  # 加载优化器
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])  #加载学习率更新策略
        args.start_epoch = checkpoint['epoch'] + 1  #加载开始的epoch
        if args.amp:
            scaler.load_state_dict(checkpoint["scaler"])

    best_dice = 0.
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs): # 对epoch进行加载
        #训练一个epoch，每10次迭代打印一次，迭代次数为训练集的大小除以batch_size
        # 这里的train_one_epoch是我们自己写的函数，对模型进行训练，传入参数，像是模型，优化器，训练集，设备，epoch，学习率更新策略，打印频率，混合精度
        mean_loss, lr = train_one_epoch(model, optimizer, train_loader, device, epoch, num_classes,
                                        lr_scheduler=lr_scheduler, print_freq=args.print_freq, scaler=scaler)
        # 对验证集进行验证，evaluate是我们自己写的函数，对模型进行验证，传入参数，像是模型，验证集，设备，类别数
        confmat, dice = evaluate(model, val_loader, device=device, num_classes=num_classes)
        val_info = str(confmat)
        print(val_info)
        print(f"dice coefficient: {dice:.3f}")
        # write into txt
        with open(results_file, "a") as f:
            # 记录每个epoch对应的train_loss、lr以及验证集各指标
            train_info = f"[epoch: {epoch}]\n" \
                         f"train_loss: {mean_loss:.4f}\n" \
                         f"lr: {lr:.6f}\n" \
                         f"dice coefficient: {dice:.3f}\n"
            f.write(train_info + val_info + "\n\n")

        if args.save_best is True:
            if best_dice < dice:
                best_dice = dice
            else:
                continue

        save_file = {"model": model.state_dict(),
                     "optimizer": optimizer.state_dict(),
                     "lr_scheduler": lr_scheduler.state_dict(),
                     "epoch": epoch,
                     "args": args}  # 保存文件
        if args.amp:
            save_file["scaler"] = scaler.state_dict()

        if args.save_best is True:
            torch.save(save_file, "save_weights/best_model.pth")
        else:
            torch.save(save_file, "save_weights/model_{}.pth".format(epoch))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("training time {}".format(total_time_str))


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="pytorch unet training")

    parser.add_argument("--data-path", default="./", help="DRIVE root")
    # exclude background
    parser.add_argument("--num-classes", default=1, type=int)
    parser.add_argument("--device", default="cuda", help="training device")
    parser.add_argument("-b", "--batch-size", default=4, type=int)
    parser.add_argument("--epochs", default=200, type=int, metavar="N",
                        help="number of total epochs to train")

    parser.add_argument('--lr', default=0.01, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--print-freq', default=1, type=int, help='print frequency')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--save-best', default=True, type=bool, help='only save best dice weights')
    # Mixed precision training parameters
    parser.add_argument("--amp", default=False, type=bool,
                        help="Use torch.cuda.amp for mixed precision training")

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()

    if not os.path.exists("./save_weights"):
        os.mkdir("./save_weights")

    main(args)
