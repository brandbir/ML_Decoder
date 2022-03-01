import os
import argparse
import time

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data.distributed

from src_files.helper_functions.bn_fusion import fuse_bn_recursively
from src_files.models import create_model
import matplotlib

from src_files.models.tresnet.tresnet import InplacABN_to_ABN

matplotlib.use('TkAgg')
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import tools
import pandas as pd
import cv2

parser = argparse.ArgumentParser(description='PyTorch MS_COCO infer')
parser.add_argument('--num-classes', default=80, type=int)
parser.add_argument('--model-path', type=str, default='./models_local/TRresNet_L_448_86.6.pth')
parser.add_argument('--pic-path', type=str, default='./pics/000000000885.jpg')
parser.add_argument('--model-name', type=str, default='tresnet_l')
parser.add_argument('--image-size', type=int, default=448)
# parser.add_argument('--dataset-type', type=str, default='MS-COCO')
parser.add_argument('--th', type=float, default=0.75)
parser.add_argument('--top-k', type=float, default=100)
# ML-Decoder
parser.add_argument('--use-ml-decoder', default=1, type=int)
parser.add_argument('--num-of-groups', default=-1, type=int)  # full-decoding
parser.add_argument('--decoder-embedding', default=768, type=int)
parser.add_argument('--zsl', default=0, type=int)

def main():   
    print('Generating detections...')

    # parsing args
    args = parser.parse_args()
    # Setup model from checkpoint
    if 'ckpt' in args.model_path:
        mode = 'ckpt'
        checkpoint = torch.load(args.model_path)
        model = create_model(args, load_head=True, from_ckpt=True).cuda()
        model.load_state_dict(checkpoint)
        model.eval()
        
    else:
        # Setup model
        mode = 'pt'
        print('creating model {}...'.format(args.model_name))
        model = create_model(args, load_head=True).cuda()
        state = torch.load(args.model_path, map_location='cpu')
        model.load_state_dict(state['model'], strict=True)
    
    ########### eliminate BN for faster inference ###########
    model = model.cpu()
    model = InplacABN_to_ABN(model)
    model = fuse_bn_recursively(model)
    model = model.cuda().half().eval()
    #######################################################
    print('model has been loaded.')

    root = '/home/brandon/Documents/datasets/mscoco'
    ds_split = 'test'

    print('loading ' + ds_split + ' dataset...')
    ds = tools.KarpathySplits(root, ds_split)
    print(len(ds), 'images were loaded.')

    if mode == 'ckpt':
        classes_list = np.array(ds.keywords)
    else:
        classes_list = np.array(list(state['idx_to_class'].values()))
    
    print('loading keywords intersected with mscoco..')
    classes_mscoco_path = '/home/brandon/Documents/git/phd/phd-image-captioning/keywords-predictor/openimages/mscoco_full_intersected_openimages_dist.csv'
    classes_mscoco = pd.read_csv(classes_mscoco_path)['Description'].values
    print('mscoco_classes:', len(classes_mscoco))

    columns_list = ['img_path', 'actual_classes', 'pred_classes', 'num_actual_classes', 'num_pred_classes', 'pred_scores']
    predictions = pd.DataFrame(columns=columns_list)

    max_imgs = len(ds)
    for i, img in enumerate(ds):
        if i < max_imgs:
            target = img[1]
            img_path = img[2]

            print(i, img_path)
            # doing inference
            im = Image.open(img_path)
            
            im_resize = im.resize((args.image_size, args.image_size))
            np_img = np.array(im_resize, dtype=np.uint8)

            if len(np_img.shape) == 2:
                np_img = cv2.cvtColor(np_img, cv2.COLOR_GRAY2RGB)
            
            tensor_img = torch.from_numpy(np_img).permute(2, 0, 1).float() / 255.0  # HWC to CHW
            tensor_batch = torch.unsqueeze(tensor_img, 0).cuda().half() # float16 inference
            output = torch.squeeze(torch.sigmoid(model(tensor_batch)))
            np_output = output.cpu().detach().numpy()


            ## Top-k predictions
        
            idx_th = np_output > args.th
            
            detected_classes = classes_list[idx_th]
            scores = np_output[idx_th]

            detected_mscoco_classes = []
            scores_mscoco = []


            for i, d in enumerate(detected_classes):
                d = d.lower()
                if d in classes_mscoco:
                    detected_mscoco_classes += [d]
                    scores_mscoco += [scores[i]]

            #print(detected_mscoco_classes)
            #print(scores_mscoco)

            scores_idx = np.argsort(-np.array(scores_mscoco))
            #print(np.array(scores_mscoco)[scores_idx])
            actual_classes = tools.convert_target_to_keywords(target, ds.keywords)
            detected_classes = list(np.array(detected_mscoco_classes)[scores_idx][:args.top_k])
            predictions = predictions.append(pd.DataFrame([[img_path, actual_classes, detected_classes, len(actual_classes), len(detected_classes), scores_mscoco]], columns=columns_list))

            # displaying image
            # print('showing image on screen...')
            # fig = plt.figure()
            # plt.imshow(im)
            # plt.axis('off')
            # plt.axis('tight')
            # plt.rcParams["axes.titlesize"] = 10
            # plt.title("detected classes: {}".format(detected_classes))

            # plt.show()
            # print('done\n')
        else:
            break

    predictions['idx'] = range(0, len(predictions)-1)
    predictions.to_csv('ml_predictions_' + ds_split + '.csv', index=False)

    


if __name__ == '__main__':
    main()


#python infer_2.py --num-classes=9605 --model-name=tresnet_m --model-path=./models_zoo/tresnet_m_open_images_200_groups_86_8.pth --num-of-groups=200 --image-size=224
#python infer_2.py --num-classes=1000 --model-name=tresnet_m --model-path=./models/openimages_b_64_labels_1000_g_4_lr_2e-4_train_decoder_grps_200/model-highest.ckpt --num-of-groups=200 --image-size=224
#python infer_2.py --num-classes=3940 --model-name=tresnet_m --model-path=./models/openimages_b_64_labels_3940_g_4_lr_2e-4_train_decoder_grps_200/model-highest.ckpt --num-of-groups=200 --image-size=224