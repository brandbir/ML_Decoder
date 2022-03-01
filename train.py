import os
import argparse
import tools
import logging
import numpy as np
import pandas as pd

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data.distributed
import torchvision.transforms as transforms
from torch.optim import lr_scheduler
from src_files.helper_functions.helper_functions import mAP, CocoDetection, CutoutPIL, ModelEma, \
    add_weight_decay
from src_files.models import create_model
from src_files.loss_functions.losses import AsymmetricLoss
from randaugment import RandAugment
from torch.cuda.amp import GradScaler, autocast

from torch_lr_finder import LRFinder



parser = argparse.ArgumentParser(description='PyTorch MS_COCO Training')
parser.add_argument('--data', type=str, default='/home/MSCOCO_2014/')
parser.add_argument('--lr', default=2e-4, type=float) # changed from 1e-4 to 2e-4
parser.add_argument('--model-name', default='tresnet_l')
parser.add_argument('--model-path', default='https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/ML_Decoder/tresnet_l_pretrain_ml_decoder.pth', type=str)
parser.add_argument('--num-classes', default=80, type=int)
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers')
parser.add_argument('--image-size', default=448, type=int,
                    metavar='N', help='input image size (default: 448)')
parser.add_argument('--batch-size', default=56, type=int,
                    metavar='N', help='mini-batch size')

parser.add_argument('--use-keywords', default=1, type=int, help="1 for MSCOCO keywords or 0 for MSCOCO objects")

# ML-Decoder
parser.add_argument('--use-ml-decoder', default=1, type=int)
parser.add_argument('--num-of-groups', default=-1, type=int)  # full-decoding
parser.add_argument('--decoder-embedding', default=768, type=int)
parser.add_argument('--zsl', default=0, type=int)

def main():
    Log_Format = "%(levelname)s %(asctime)s - %(message)s"

    logging.basicConfig(filename = "train.log",
                    filemode = "w",
                    format = Log_Format, 
                    level = logging.INFO)

    logger = logging.getLogger()

    args = parser.parse_args()

    # COCO Data loading
    instances_path_val = os.path.join(args.data, 'annotations/instances_val2014.json')
    instances_path_train = os.path.join(args.data, 'annotations/instances_train2014.json')
    # data_path_val = args.data
    # data_path_train = args.data
    data_path_val = f'{args.data}/val2014'  # args.data
    data_path_train = f'{args.data}/train2014'  # args.data
    
    if args.use_keywords == 1:
        print('Loading from Karpathy splits...')
        val_dataset = tools.KarpathySplits(args.data,
                                    'val',
                                    transforms.Compose([
                                        transforms.Resize((args.image_size, args.image_size)),
                                        transforms.ToTensor(),
                                        # normalize, # no need, toTensor does normalization
                                    ]))

        train_dataset = tools.KarpathySplits(args.data,
                                    'train',
                                    transforms.Compose([
                                        transforms.Resize((args.image_size, args.image_size)),
                                        CutoutPIL(cutout_factor=0.5),
                                        RandAugment(),
                                        transforms.ToTensor(),
                                        # normalize,
                                    ]))
    elif args.use_keywords == 0:
        print('Loading from Coco detections...')
        val_dataset = CocoDetection(data_path_val,
                                instances_path_val,
                                transforms.Compose([
                                    transforms.Resize((args.image_size, args.image_size)),
                                    transforms.ToTensor(),
                                    # normalize, # no need, toTensor does normalization
                                ]), num_classes=args.num_classes)
        train_dataset = CocoDetection(data_path_train,
                                    instances_path_train,
                                    transforms.Compose([
                                        transforms.Resize((args.image_size, args.image_size)),
                                        CutoutPIL(cutout_factor=0.5),
                                        RandAugment(),
                                        transforms.ToTensor(),
                                        # normalize,
                                    ]), num_classes=args.num_classes)

    else:
        exit('invalid --keywords argument.')

    print("len(val_dataset): ", len(val_dataset))
    print("len(train_dataset): ", len(train_dataset))
    
    if args.use_keywords:
        print("number of keywords:", len(train_dataset.keywords))

    # print("len(val_dataset)): ", len(val_dataset.images))
    # print("len(train_dataset)): ", len(train_dataset.images))
    
    # Setup model
    print('creating model {}...'.format(args.model_name))
    model = create_model(args).cuda()
    print('done')

    # Pytorch Data loader
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False)

    #lr_finder(model, train_loader, val_loader)

    #Actuall Training
    train_multi_label_coco(model, train_loader, val_loader, args.lr, logger, args.use_keywords)

def lr_finder(model, train_loader, val_loader):
    lr = 1e-7
    weight_decay = 1e-4
    criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=0, clip=0.05, disable_torch_grad_focal_loss=True)
    parameters = add_weight_decay(model, weight_decay)
    optimizer = torch.optim.Adam(params=parameters, lr=lr, weight_decay=0)
    #optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    lr_finder = LRFinder(model, optimizer, criterion, device="cuda")
    lr_finder.range_test(train_loader, end_lr=1, num_iter=100)
    lr_finder.plot(log_lr=False)
    lr_finder.reset()




def train_multi_label_coco(model, train_loader, val_loader, lr, logger, use_keywords=True):
    pd_training_results = pd.DataFrame(columns=['epoch', 'epoch_mean_loss', 'val_maP_score'])
    
    ema = ModelEma(model, 0.9997)  # 0.9997^641=0.82

    # set optimizer
    Epochs = 40
    weight_decay = 1e-4
    criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=0, clip=0.05, disable_torch_grad_focal_loss=True)
    parameters = add_weight_decay(model, weight_decay)
    optimizer = torch.optim.Adam(params=parameters, lr=lr, weight_decay=0)  # true wd, filter_bias_and_bn
    steps_per_epoch = len(train_loader)
    scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=steps_per_epoch, epochs=Epochs,
                                        pct_start=0.2)

    highest_mAP = 0
    trainInfoList = []
    scaler = GradScaler()
    
    for epoch in range(Epochs):
        total_batch_losses = 0

        for i, instance in enumerate(train_loader):
            inputData = instance[0]
            target =  instance[1]
            inputData = inputData.cuda()
            target = target.cuda()  # (batch,3,num_classes)
            if not use_keywords:
                target = target.max(dim=1)[0]

            with autocast():  # mixed precision
                output = model(inputData).float()  # sigmoid will be done in loss !
            loss = criterion(output, target)
            
            total_batch_losses += loss.item()

            model.zero_grad()            
            scale = scaler.get_scale()
            scaler.scale(loss).backward()
            # loss.backward()

            scaler.step(optimizer)
            scaler.update()
            #optimizer.step()
            
            skip_sched = (scale != scaler.get_scale())
            
            if not skip_sched:
                scheduler.step()

            ema.update(model)
            
            # store information
            if i % 100 == 0:
                trainInfoList.append([epoch, i, loss.item()])
                msg = 'Epoch [{}/{}], Step [{}/{}], LR {:.1e}, Loss: {:.5f}'.format(epoch, Epochs, str(i).zfill(3), str(steps_per_epoch).zfill(3),
                            scheduler.get_last_lr()[0], \
                            loss.item())
                print(msg)
                logger.info(msg)

        try:
            torch.save(model.state_dict(), os.path.join(
                'models/', 'model-{}-{}.ckpt'.format(epoch + 1, i + 1)))
        except:
            pass

        model.eval()

        mAP_score = validate_multi(val_loader, model, ema, use_keywords)
        model.train()
        if mAP_score > highest_mAP:
            highest_mAP = mAP_score
            try:
                torch.save(model.state_dict(), os.path.join(
                    'models/', 'model-highest.ckpt'))
            except:
                pass
        
        mean_batch_loss = total_batch_losses/float(len(train_loader))

        row = pd.DataFrame([[epoch, mean_batch_loss, mAP_score]], columns=pd_training_results.columns)
        pd_training_results = pd.concat([pd_training_results,row]).reset_index(drop=True)
        pd_training_results.to_csv('training_results.csv', index=False)

        msg = 'current_mAP = {:.2f}, highest_mAP = {:.2f}, Average Loss: {:.5f}\n'.format(mAP_score, highest_mAP, mean_batch_loss)
        print(msg)
        logger.info(msg)


def validate_multi(val_loader, model, ema_model, use_keywords):
    print("starting validation")
    Sig = torch.nn.Sigmoid()
    preds_regular = []
    preds_ema = []
    targets = []
    for i, instance in enumerate(val_loader):
        input = instance[0]
        target = instance[1]
        #target = target
        if not use_keywords:
            target = target.max(dim=1)[0]
        # compute output
        with torch.no_grad():
            with autocast():
                output_regular = Sig(model(input.cuda())).cpu()
                output_ema = Sig(ema_model.module(input.cuda())).cpu()

        # for mAP calculation
        preds_regular.append(output_regular.cpu().detach())
        preds_ema.append(output_ema.cpu().detach())
        targets.append(target.cpu().detach())

    mAP_score_regular = mAP(torch.cat(targets).numpy(), torch.cat(preds_regular).numpy())
    mAP_score_ema = mAP(torch.cat(targets).numpy(), torch.cat(preds_ema).numpy())
    print("mAP score regular {:.2f}, mAP score EMA {:.2f}".format(mAP_score_regular, mAP_score_ema))
    return max(mAP_score_regular, mAP_score_ema)


if __name__ == '__main__':
    main()

# python train.py --data=/home/brandon/Documents/datasets/mscoco/ --model-name=tresnet_m --image-size=224 --model-path=models_zoo/tresnet_m_COCO_224_84_2.pth --use-keywords=0
# python train.py --data=/home/brandon/Documents/datasets/mscoco/ --model-name=tresnet_l --num-classes=1000 --image-size=448 --batch-size=32 --num-of-groups=10
# python train.py --data=/home/brandon/Documents/datasets/mscoco/ --model-name=tresnet_m --num-classes=1000 --image-size=224 --batch-size=64 --model-path=models_zoo/tresnet_m_open_images_200_groups_86_8.pth
# python train.py --data=/home/brandon/Documents/datasets/mscoco/ --model-name=tresnet_m --num-classes=3940 --image-size=224 --batch-size=64 --model-path=models_zoo/tresnet_m_open_images_200_groups_86_8.pth --num-of-groups=200

