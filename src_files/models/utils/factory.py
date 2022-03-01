import logging
import os
from urllib import request

import torch

from ...ml_decoder.ml_decoder import add_ml_decoder_head

logger = logging.getLogger(__name__)

from ..tresnet import TResnetM, TResnetL, TResnetXL

from torchinfo import summary



def create_model(args,load_head=False, from_ckpt=False):
    """Create a model
    """
    model_params = {'args': args, 'num_classes': args.num_classes}
    args = model_params['args']
    args.model_name = args.model_name.lower()

    if args.model_name == 'tresnet_m':
        model = TResnetM(model_params)
    elif args.model_name == 'tresnet_l':
        model = TResnetL(model_params)
    elif args.model_name == 'tresnet_xl':
        model = TResnetXL(model_params)
    else:
        print("model: {} not found !!".format(args.model_name))
        exit(-1)

    ####################################################################################
    if args.use_ml_decoder:
        model = add_ml_decoder_head(model,num_classes=args.num_classes,num_of_groups=args.num_of_groups,
                                    decoder_embedding=args.decoder_embedding, zsl=args.zsl)
    ####################################################################################
    #summary(model)
    # loading pretrain model
    model_path = args.model_path
    if args.model_name == 'tresnet_l' and os.path.exists("./tresnet_l.pth"):
        model_path = "./tresnet_l.pth"
    if model_path:  # make sure to load pretrained model
        if not os.path.exists(model_path):
            print("downloading pretrain model...")
            request.urlretrieve(args.model_path, "./tresnet_l.pth")
            model_path = "./tresnet_l.pth"
            print('done')
        state = torch.load(model_path, map_location='cpu')
        
        if not load_head:
            filtered_dict = {k: v for k, v in state['model'].items() if
                             (k in model.state_dict() and 'head.fc' not in k)}

            filtered_dict = {k: v for k, v in state['model'].items() if
                             (k in model.state_dict() and 'head.decoder.duplicate_pooling' not in k
                             and 'head.decoder.query_embed.weight' not in k)}

            # filtered_dict = {k: v for k, v in state['model'].items() if
            #                  (k in model.state_dict() and 'head.decoder' not in k)}

            model.load_state_dict(filtered_dict, strict=False)
        elif not from_ckpt:
            model.load_state_dict(state['model'], strict=True)


    # FREEZING HEAD and TResNet layer 4
    # for name, param in model.named_parameters():
    #     if not ('head.decoder' in name): # or 'body.layer4' in name
    #         param.requires_grad=False

    # for name, param in model.named_parameters():
    #     print(name, param.requires_grad)

    #summary(model)
    #exit()
    
    return model
