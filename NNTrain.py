import os
import errno
import torch
import timeit

import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.functional as F
from torch.utils import data

# import Image
# from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import cosine_distances
from scipy import spatial
from sklearn.metrics import mean_squared_error
from torch.optim import lr_scheduler
from NNLoss import dice_loss
from NNMetrics import segmentation_scores, f1_score, hd95
from NNUtils import CustomDataset, evaluate, test
from tensorboardX import SummaryWriter
from adamW import AdamW
from torch.autograd import Variable

from NNBaselines import GCNonLocal_UNet_All
from NNBaselines import UNet
from NNBaselines import CBAM_UNet_All
from NNBaselines import DilatedUNet
from NNBaselines import AttentionUNet
from NNBaselines import CSE_UNet_Full, Deeper_CSE_UNet_Full

from NNBaselines import FCN8, DeeperUNet

from NNLoss import f_beta_loss

def trainModels(
                data_directory,
                dataset_name,
                input_dim,
                class_no,
                repeat,
                train_batchsize,
                validate_batchsize,
                num_epochs,
                learning_rate,
                width,
                network,
                lr_decay=True,
                augmentation=True,
                reverse=False):

    for j in range(1, repeat + 1):

        repeat_str = str(j)

        # ==================================================
        # Baselines
        # ==================================================

        if network == 'unet':
            assert reverse is False
            Exp = UNet(in_ch=input_dim, width=width, class_no=class_no)
            Exp_name = 'UNet_batch_' + str(train_batchsize) + \
                       '_width_' + str(width) + \
                       '_repeat_' + repeat_str + \
                       '_augment_' + str(augmentation) + \
                       '_lr_decay_' + str(lr_decay)

        elif network == 'dilated_unet':
            assert reverse is False
            dilation = 9
            Exp = DilatedUNet(in_ch=input_dim, width=width, dilation=dilation)
            Exp_name = 'DilatedUNet_batch_' + str(train_batchsize) + \
                       '_width_' + str(width) + \
                       '_dilation_' + str(dilation) + \
                       '_repeat_' + repeat_str + \
                       '_augment_' + str(augmentation) + \
                       '_lr_decay_' + str(lr_decay)

        elif network == 'deeper_unet':
            assert reverse is False
            Exp = DeeperUNet(in_ch=input_dim, width=width)
            Exp_name = 'DeeperUnet_' + \
                       '_width_' + str(width) + \
                       '_repeat_' + repeat_str + \
                       '_augment_' + str(augmentation) + \
                       '_lr_decay_' + str(lr_decay)

        elif network == 'fcn':
            assert reverse is False
            Exp = FCN8(in_ch=input_dim, width=width)
            Exp_name = 'FCN8s_' + \
                       '_width_' + str(width) + \
                       '_repeat_' + repeat_str + \
                       '_augment_' + str(augmentation) + \
                       '_lr_decay_' + str(lr_decay)

        elif network == 'atten_unet':
            assert reverse is False
            Exp = AttentionUNet(in_ch=input_dim, width=width)
            Exp_name = 'AttentionUNet_batch_' + str(train_batchsize) + \
                       '_Valbatch_' + str(validate_batchsize) + \
                       '_width_' + str(width) + \
                       '_repeat_' + repeat_str + \
                       '_augment_' + str(augmentation) + \
                       '_lr_decay_' + str(lr_decay)

        elif network == 'cse_unet_full':
            # assert visualise_attention is True
            assert reverse is False
            # didn't have time to write the code to visulisae attention weights for cse u net
            Exp = CSE_UNet_Full(in_ch=input_dim, width=width)
            Exp_name = 'CSEUNetFull_batch_' + str(train_batchsize) + \
                       '_Valbatch_' + str(validate_batchsize) + \
                       '_width_' + str(width) + \
                       '_repeat_' + repeat_str + \
                       '_augment_' + str(augmentation) + \
                       '_lr_decay_' + str(lr_decay)

        elif network == 'deeper_cse_unet_full':
            # assert visualise_attention is True
            assert reverse is False
            # didn't have time to write the code to visulisae attention weights for cse u net
            Exp = Deeper_CSE_UNet_Full(in_ch=input_dim, width=width)
            Exp_name = 'DeeperCSEUNetFull_batch_' + str(train_batchsize) + \
                       '_Valbatch_' + str(validate_batchsize) + \
                       '_width_' + str(width) + \
                       '_repeat_' + repeat_str + \
                       '_augment_' + str(augmentation) + \
                       '_lr_decay_' + str(lr_decay)

        else:
            print('Using default network: U-net.')
            Exp = UNet(in_ch=input_dim, width=width, class_no=class_no)
            Exp_name = 'UNet_batch_' + str(train_batchsize) + \
                       '_width_' + str(width) + \
                       '_repeat_' + repeat_str + \
                       '_augment_' + str(augmentation) + \
                       '_lr_decay_' + str(lr_decay)

        # ====================================================================================================================================================================
        trainloader, validateloader, testloader, train_dataset, validate_dataset, test_dataset = getData(data_directory, dataset_name, train_batchsize, validate_batchsize, augmentation)
        # ===================
        trainSingleModel(Exp,
                         Exp_name,
                         num_epochs,
                         learning_rate,
                         dataset_name,
                         train_dataset,
                         train_batchsize,
                         trainloader,
                         validateloader,
                         testloader,
                         reverse_mode=reverse,
                         lr_schedule=lr_decay,
                         class_no=class_no)


def getData(data_directory, dataset_name, train_batchsize, validate_batchsize, data_augment):

    train_image_folder = data_directory + dataset_name + '/train/patches'
    train_label_folder = data_directory + dataset_name + '/train/labels'
    validate_image_folder = data_directory + dataset_name + '/validate/patches'
    validate_label_folder = data_directory + dataset_name + '/validate/labels'
    test_image_folder = data_directory + dataset_name + '/test/patches'
    test_label_folder = data_directory + dataset_name + '/test/labels'

    # print(train_image_folder)

    train_dataset = CustomDataset(train_image_folder, train_label_folder, data_augment)

    validate_dataset = CustomDataset(validate_image_folder, validate_label_folder, 'full')

    test_dataset = CustomDataset(test_image_folder, test_label_folder, 'full')

    trainloader = data.DataLoader(train_dataset, batch_size=train_batchsize, shuffle=True, num_workers=1, drop_last=True)

    validateloader = data.DataLoader(validate_dataset, batch_size=validate_batchsize, shuffle=False, num_workers=1, drop_last=False)

    testloader = data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1, drop_last=False)

    return trainloader, validateloader, testloader, train_dataset, validate_dataset, test_dataset

# =====================================================================================================================================


def trainSingleModel(model,
                     model_name,
                     num_epochs,
                     learning_rate,
                     datasettag,
                     train_dataset,
                     train_batchsize,
                     trainloader,
                     validateloader,
                     testdata,
                     reverse_mode,
                     lr_schedule,
                     class_no):

    # change log names
    training_amount = len(train_dataset)

    iteration_amount = training_amount // train_batchsize

    iteration_amount = iteration_amount - 1

    device = torch.device('cuda')

    lr_str = str(learning_rate)

    epoches_str = str(num_epochs)

    save_model_name = model_name + '_' + datasettag + '_e' + epoches_str + '_lr' + lr_str

    saved_information_path = './Results'
    try:
        os.mkdir(saved_information_path)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass

    saved_information_path = saved_information_path + '/' + save_model_name
    try:
        os.mkdir(saved_information_path)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass

    saved_model_path = saved_information_path + '/trained_models'
    try:
        os.mkdir(saved_model_path)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass

    print('The current model is:')
    print(save_model_name)
    print('\n')

    writer = SummaryWriter('./Results/Log_' + datasettag + '/' + save_model_name)

    model.to(device)

    threshold = torch.tensor([0.5], dtype=torch.float32, device=device, requires_grad=False)
    upper = torch.tensor([1.0], dtype=torch.float32, device=device, requires_grad=False)
    lower = torch.tensor([0.0], dtype=torch.float32, device=device, requires_grad=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-5)

    if lr_schedule is True:

        # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.2, patience=10, threshold=0.001)
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[num_epochs // 2, 3*num_epochs // 4], gamma=0.1)

    start = timeit.default_timer()

    for epoch in range(num_epochs):

        model.train()
        train_iou = []
        train_loss = []

        # j: index of iteration
        for j, (images, labels, imagename) in enumerate(trainloader):

            optimizer.zero_grad()

            images = images.to(device=device, dtype=torch.float32)

            if class_no == 2:
                labels = labels.to(device=device, dtype=torch.float32)
            else:
                labels = labels.to(device=device, dtype=torch.long)

            outputs = model(images)

            if class_no == 2:
                prob_outputs = torch.sigmoid(outputs)
                loss = dice_loss(prob_outputs, labels)
                class_outputs = torch.where(prob_outputs > threshold, upper, lower)
            else:
                prob_outputs = torch.softmax(outputs, dim=1)
                # loss = nn.CrossEntropyLoss(reduction='mean')(prob_outputs, labels)
                loss = nn.CrossEntropyLoss(reduction='mean')(prob_outputs, labels.squeeze(1))
                _, class_outputs = torch.max(outputs, dim=1)

            loss.backward()
            optimizer.step()

            mean_iu_, _, __ = segmentation_scores(labels, class_outputs, class_no)
            train_iou.append(mean_iu_)
            train_loss.append(loss.item())

        if lr_schedule is True:
            # scheduler.step(validate_iou)
            scheduler.step()
        else:
            pass

        model.eval()

        with torch.no_grad():

            validate_iou = []
            validate_f1 = []
            validate_h_dist = []

            for i, (val_images, val_label, imagename) in enumerate(validateloader):

                val_img = val_images.to(device=device, dtype=torch.float32)

                if class_no == 2:
                    val_label = val_label.to(device=device, dtype=torch.float32)
                else:
                    val_label = val_label.to(device=device, dtype=torch.long)

                assert torch.max(val_label) != 100.0

                val_outputs = model(val_img)
                if class_no == 2:
                    val_class_outputs = torch.sigmoid(val_outputs)
                    val_class_outputs = (val_class_outputs > 0.5).float()
                else:
                    val_class_outputs = torch.softmax(val_outputs, dim=1)
                    _, val_class_outputs = torch.max(val_class_outputs, dim=1)

                # b, c, h, w = val_label.size()
                # val_class_outputs = val_class_outputs.reshape(b, c, h, w)

                eval_mean_iu_, _, __ = segmentation_scores(val_label, val_class_outputs, class_no)
                eval_f1_, eval_recall_, eval_precision_, eTP, eTN, eFP, eFN, eP, eN = f1_score(val_label, val_class_outputs, class_no)

                validate_iou.append(eval_mean_iu_)
                validate_f1.append(eval_f1_)

                if (val_class_outputs == 1).sum() > 1 and (val_label == 1).sum() > 1 and class_no == 2:
                    v_dist_ = hd95(val_class_outputs, val_label, class_no)
                    validate_h_dist.append(v_dist_)

        print(
            'Step [{}/{}], '
            'Train loss: {:.4f}, '
            'Train iou: {:.4f}, '
            'val iou:{:.4f}, '.format(epoch + 1, num_epochs,
                                      np.nanmean(train_loss),
                                      np.nanmean(train_iou),
                                      np.nanmean(validate_iou)))

        writer.add_scalars('acc metrics', {'train iou': np.nanmean(train_iou),
                                           'val iou': np.nanmean(validate_iou),
                                           'val f1': np.nanmean(validate_f1)}, epoch + 1)

        if epoch > num_epochs - 10:

            save_model_name_full = saved_model_path + '/epoch' + str(epoch)
            save_model_name_full = save_model_name_full + '.pt'
            path_model = save_model_name_full
            torch.save(model, path_model)

    test(testdata,
         saved_model_path,
         device,
         reverse_mode=reverse_mode,
         class_no=class_no,
         save_path=saved_model_path)

    # save model
    stop = timeit.default_timer()
    print('Time: ', stop - start)
    print('\nTraining finished and model saved\n')

    return model

