import torch
import torch.nn as nn
import os
import random
import logging
import numpy as np
import torch.nn.functional as F
import cv2

from torchvision import transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold

from utils import ContextLoss, LocalLoss, CosineAnnealingWarmRestarts
from Load_Dataset import RandomGenerator, ValGenerator, KfoldPairDataset, show_dataset
import config.Config_MF_cv as config


def logger_config(log_path):
    loggerr = logging.getLogger()
    loggerr.setLevel(level=logging.INFO)
    handler = logging.FileHandler(log_path, encoding='UTF-8')
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    loggerr.addHandler(handler)
    loggerr.addHandler(console)
    return loggerr


def save_checkpoint(state, save_path):
    '''
        Save the current model.
        If the model is the best model since beginning of the training
        it will be copy
    '''
    logger.info('\t Saving to {}'.format(save_path))
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    epoch = state['epoch']  # epoch no
    best_model = state['best_model']  # bool
    model = state['model']  # model type

    if best_model:
        filename = save_path + '/' + \
                   'best_model-{}.pth.tar'.format(model)
    else:
        filename = save_path + '/' + \
                   'model-{}-{:02d}.pth.tar'.format(model, epoch)
    torch.save(state, filename)


def save_on_batch(images1, masks, pred, names, vis_path):
    '''Computes the mean Area Under ROC Curve over a batch during training'''
    if pred.shape[1] != 1:
        pred = torch.argmax(pred.detach(), dim=1).cpu().numpy()
        if len(masks.shape) == 4:
            masks = torch.argmax(masks.detach(), dim=1).cpu().numpy()
        else:
            masks = masks.cpu().numpy()

    for i in range(pred.shape[0]):
        if pred.shape[1] == 1:
            pred_tmp = pred[i][0].cpu().detach().numpy()
            mask_tmp = masks[i].cpu().detach().numpy()
            pred_tmp[pred_tmp >= 0.5] = 255
            pred_tmp[pred_tmp < 0.5] = 0
            mask_tmp[mask_tmp > 0] = 255
            mask_tmp[mask_tmp <= 0] = 0
        else:
            pred_tmp = pred[i] * 127
            mask_tmp = masks[i] * 127

        cv2.imwrite(vis_path + names[i][:-4] + "_pred.jpg", pred_tmp)
        cv2.imwrite(vis_path + names[i][:-4] + "_gt.jpg", mask_tmp)


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    # 也可以判断是否为conv2d，使用相应的初始化方式
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    # 是否为批归一化层
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


def train_one_epoch(model, train_loader, logger, optimizer, loss_fn1, loss_fn2, lr_scheduler, device):
    total_loss2_sum, total_loss1_sum = 0., 0.
    dice_local_sum, dice_global_sum = 0., 0.
    num_images = len(train_loader.dataset)
    for i, (sampled_batch, names) in enumerate(train_loader, 1):
        cell_img, cell_mask = sampled_batch['cell_img'].to(device), sampled_batch['cell_mask'].to(device)
        tis_img, tis_mask = sampled_batch['tis_img'].to(device), sampled_batch['tis_mask'].to(device)
        pos = sampled_batch['pos'].to(device)

        # print("-------------------")
        # print(input_data1)
        # 前向传播
        if config.model_name not in config.mutil_task_model:
            output = model(cell_img)
        else:
            output = model(cell_img, tis_img, pos)

        # # 计算损失
        # target1 = torch.ones(1,1,224, 224)
        loss1 = loss_fn1(output['x1'], cell_mask)

        # target2 = torch.tensor([[0.,1.]])
        # target2 = torch.zeros(1,1,224, 224)
        if config.model_name in config.mutil_task_model:
            # print("------------Tis_mask----------")
            # print(f"tis_mask.unique{tis_mask.unique(return_counts=True)}")
            loss2 = loss_fn2(output['x2'], tis_mask)
            loss = loss1['loss'] + loss2['loss']
        else:
            loss = loss1['loss']

        total_loss1_sum += len(cell_img) * loss1['loss']
        dice_local = 1 - loss1['dice']
        dice_local_sum += len(cell_img) * dice_local

        if config.model_name in config.mutil_task_model:
            total_loss2_sum += len(cell_img) * loss2['loss']
            dice_global = loss2['dice']
            dice_global_sum += len(cell_img) * dice_global

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    average_loss1 = total_loss1_sum / num_images
    average_dice_local = dice_local_sum / num_images
    logger.info(
        f"train average loss local:{average_loss1:.3f} train average_dice_local:{average_dice_local:.3f}" + "\n")

    if config.model_name in config.mutil_task_model:
        average_loss2 = total_loss2_sum / num_images
        average_dice_global = dice_global_sum / num_images
        logger.info(
            f"train average loss global:{average_loss2:.3f} train average_dice_global:{average_dice_global:.3f}" + "\n")

    lr_scheduler.step()
    torch.cuda.empty_cache()
    return average_loss1, average_dice_local


def val_one_epoch(model, loader, logger, loss_fn1, loss_fn2, epoch, device, visual=True):
    total_loss2_sum, total_loss1_sum = 0., 0.
    dice_local_sum, dice_global_sum = 0., 0.
    num_images = len(loader.dataset)
    for i, (sampled_batch, names) in enumerate(loader, 1):
        cell_img, cell_mask = sampled_batch['cell_img'].to(device), sampled_batch['cell_mask'].to(device)
        tis_img, tis_mask = sampled_batch['tis_img'].to(device), sampled_batch['tis_mask'].to(device)
        pos = sampled_batch['pos'].to(device)

        if config.model_name not in config.mutil_task_model:
            output = model(cell_img)
        else:
            output = model(cell_img, tis_img, pos)

        loss1 = loss_fn1(output['x1'], cell_mask)
        total_loss1_sum += len(cell_img) * loss1['loss']
        # loss_fn1._show_dice()
        dice_local = 1 - loss1['dice']
        dice_local_sum += len(cell_img) * dice_local

        if 'x2' in output.keys():
            loss2 = loss_fn2(output['x2'], tis_mask)
            total_loss2_sum += len(cell_img) * loss2['loss']
            dice_global = 1 - loss2['dice']
            dice_global_sum += len(cell_img) * dice_global

        if epoch % config.vis_frequency == 0 and visual:
            # print(names)
            vis_path = config.visualize_path + str(epoch) + '/'
            os.makedirs(vis_path, exist_ok=True)
            if random.random() < 0.2:
                if config.model_name in config.mutil_task_model:
                    vis_path_tis = config.visualize_path + str(epoch) + '_tis/' + ''
                    os.makedirs(vis_path_tis, exist_ok=True)
                    save_on_batch(cell_img, cell_mask, output['x1'], names, vis_path)
                    save_on_batch(tis_img, tis_mask, output['x2'], names, vis_path_tis)
                else:
                    save_on_batch(cell_img, cell_mask, output['x1'], names, vis_path)

    average_dice_local = dice_local_sum / num_images
    average_dice_global = dice_global_sum / num_images
    average_loss1 = total_loss1_sum / num_images
    average_loss2 = total_loss2_sum / num_images

    logger.info(f"val average loss_local:{average_loss1:.3f}  val average_dice_local:{average_dice_local:.3f}" + "\n")
    if 'x2' in output.keys():
        logger.info(
            f"val average_loss_global:{average_loss2:.3f} val average_dice_global:{average_dice_global:.3f}" + "\n")
    torch.cuda.empty_cache()

    return average_loss1, average_dice_local


def main_loop(logger, train_loader, val_loader, test_loader, model_type='', fold=0, kfold=5):
    device = torch.device('cuda:1' if torch.cuda.is_available() else "cpu")

    logger.info(f"Model:{config.model_name} task name {config.task_name}")
    if model_type == 'DoubleDeepLab':
        from nets.DoubleNet import DoubleDeepLab
        model = DoubleDeepLab(cell_classes=config.local_classes, tis_classes=config.context_classes).to(device)

    elif model_type == "DeepLab":
        from nets.deeplabv3 import DeepLab
        model = DeepLab(num_classes=config.n_labels).to(device)

    elif config.model_name == "MFCTnet":
        from nets.MFCT import MFCTnet
        if config.backbone == "mobilenet":
            model = MFCTnet(config, num_classes=3,backbone='mobilenet').to(device)
        elif config.backbone == "xception":
            model = MFCTnet(config, num_classes=3,backbone='xception').to(device)
        elif config.backbone == "resnet34":
            model = MFCTnet(config, num_classes=3,backbone='resnet34').to(device)
        else:
            print("请输入正确的模型名字")
            exit()
    else:
        print('请输入正确的模型名字')
        exit()

    model.apply(weight_init)
    # 创建输入数据

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)

    lr_scheduler = CosineAnnealingLR(optimizer, eta_min=1e-6, T_max=config.epochs)

    # 创建损失函数
    # loss_fn1 = nn.MSELoss()
    # loss_fn2 = nn.MSELoss()
    # ocelot样本类别数{0: 19215144, 1: 213341, 2: 391035}，用最大样本数除各个类别样本得到权重

    loss_local = LocalLoss(mode="multiclass")
    loss_context = ContextLoss(mode="binary", ignore_index=254)

    min_loss = 1000.
    max_dice = 0.0
    best_epoch = 1
    min_loss_epoch = 1
    best_test_dice = 0.0

    start_epoch = 0
    if config.resume:
        print('------------resume-----------------')
        path_checkpoint = config.check_point
        checkpoint = torch.load(path_checkpoint)

        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        # train_criterion.load_state_dict(checkpoint['criterion_state_dict'])
        start_epoch = checkpoint['epoch']
        print("start_epoch:", start_epoch)
        print('-----------------------------')


    for epoch in range(start_epoch, config.epochs + 1):
        logger.info('\n=========Fold[{}/{}] Epoch [{}/{}] ========='.format(fold, kfold, epoch + 1, config.epochs + 1))
        logger.info(config.session_name)
        model.train(True)

        train_loss_local, train_dice_local = train_one_epoch(model, train_loader, logger, optimizer, loss_local,
                                                             loss_context, lr_scheduler, device)
        logger.info('Validation')
        with torch.no_grad():
            model.eval()
            val_loss, epoch_dice = val_one_epoch(model, val_loader, logger, loss_local, loss_context, epoch, device)

        if train_loss_local < min_loss:
            min_loss_epoch = epoch
            logger.info(f'Min loss decrease from {min_loss} to {train_loss_local}' + '\n')
            min_loss = train_loss_local
        else:
            logger.info(
                f'Min loss {train_loss_local} does not decrease, the min is still {min_loss} in {min_loss_epoch}' + '\n')

        if epoch_dice > max_dice:
            if epoch + 1 > 5:
                logger.info(f'Best dice increased from {max_dice:.4f} to {epoch_dice:.4f}')
                max_dice = epoch_dice
                best_epoch = epoch + 1
                save_checkpoint({'epoch': epoch,
                                 'best_model': True,
                                 'model': model_type,
                                 'state_dict': model.state_dict(),
                                 'val_loss': val_loss,
                                 'optimizer': optimizer.state_dict()}, config.model_path + "fold_" + str(fold) + "/")

                with torch.no_grad():
                    model.eval()
                    test_loss, test_dice = val_one_epoch(model, test_loader, logger, loss_local, loss_context, epoch,
                                                         device, visual=False)
                    best_test_dice = test_dice
                    logger.info(
                        f'-----------------------------Best Test Dice is:{test_dice:.4f} -------------------------------------------------------' + '\n')
        else:
            logger.info(
                f'Mean dice:{epoch_dice:.4f} does not increase, the best is still: {max_dice:.4f} in epoch {best_epoch}' + '\n')

        early_stopping_count = epoch - best_epoch + 1
        logger.info('\t early_stopping_count: {}/{}'.format(early_stopping_count, config.early_stopping_patience))

        if early_stopping_count > config.early_stopping_patience:
            logger.info('\t early_stopping!')
            break

    return max_dice, best_test_dice


if __name__ == '__main__':
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    if not os.path.isdir(config.save_path):
        os.makedirs(config.save_path)
    logger = logger_config(log_path=config.logger_path)

    filelists = np.array(os.listdir(config.train_dataset + "cell"))
    test_filelist = np.array(os.listdir(config.test_dataset + "cell"))
    kfold = 5
    kf = KFold(n_splits=kfold, shuffle=True, random_state=config.seed)

    dice_list = []
    test_dice_list = []
    img_size = config.img_size
    batch_size = config.batch_size
    logger.info(f'image_size:{img_size}, batchsize:{batch_size}')

    for fold, (train_index, val_index) in enumerate(kf.split(filelists)):
        train_filelists = filelists[train_index]
        val_filelists = filelists[val_index]
        np.savetxt(config.save_path + "val_fold_" + str(fold + 1) + ".txt", val_filelists, '%s')
        logger.info(
            "Total Nums: {}, train: {}, val: {}".format(len(filelists), len(train_filelists), len(val_filelists)))

        train_tf = RandomGenerator(output_size=[config.img_size, config.img_size])

        val_tf = ValGenerator(output_size=[config.img_size, config.img_size])

        train_dataset = KfoldPairDataset(config.train_dataset, train_filelists, train_tf, image_size=config.img_size)
        val_dataset = KfoldPairDataset(config.train_dataset, val_filelists, val_tf, image_size=config.img_size)
        test_dataset = KfoldPairDataset(config.test_dataset, test_filelist, val_tf, image_size=config.img_size)
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=8,
                                  pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=True, num_workers=8, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=True, num_workers=8,
                                 pin_memory=True)

        dice, test_dice = main_loop(logger, train_loader, val_loader, test_loader, model_type=config.model_name,
                                    fold=fold + 1, kfold=kfold)
        dice_list.append(dice)
        test_dice_list.append(test_dice)

    dice = 0.0
    for j in range(len(dice_list)):
        logging.info("fold {0}: {1:2.4f}".format(j + 1, dice_list[j]))
        dice += dice_list[j]
    logger.info(f"val_dice{dice_list} \n test dice{test_dice_list}")
    logging.info("mean dice: {:.4f} \n".format(dice / kfold))

#     F1分数
    from evaluation.predict import process
    from evaluation.eval import main_eval
    from pathlib import Path
    f1scores = []
    for i in range(1,6):
        WEIGHT_PATH = Path(config.model_path+f"/fold_{i}/"+'best_model-{}.pth.tar'.format(config.model_name))
        GROUND_TRUTH_PATH = Path(f"/mnt/data1/test/workspace/MFCTnet/MFCT/evaluation/results/cell_gt_test.json")
        out_path = Path(os.path.join(os.path.dirname(WEIGHT_PATH), 'predict_test.json'))
        # print(out_path)
        process(weight_path=WEIGHT_PATH)
        # print("aaa")
        f1 = main_eval(out_path,GROUND_TRUTH_PATH)
        f1scores.append(f1)
    # for i,f1_score in enumerate(f1scores):
    #     print(f"----------------------fold_{i}-------------------------")
    #     print(f1_score)
