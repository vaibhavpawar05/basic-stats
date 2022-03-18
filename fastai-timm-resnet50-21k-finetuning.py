from fastai.vision.all import *
from fastai.vision.learner import _update_first_layer

import os
import pandas as pd
from pathlib import Path
import sys
from sklearn.model_selection import train_test_split
import gc
import shutil

import timm
import argparse

# argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("-bs", type=int)
args = parser.parse_args()

bs = args.bs

# Set the path to the photos.
path = Path('/media/manu/only_one_cropped_data_aug_resized_images/jordan1_foot_bed_2/')

torch.cuda.empty_cache()
 
# # Getting image files.
train_files = get_image_files(path/'train')
val_files = get_image_files(path/'val')

jorda1_footbed_stitching = DataBlock(blocks = (ImageBlock, CategoryBlock),
                                     get_items=get_image_files, 
                                     splitter=GrandparentSplitter(train_name='train', valid_name='val'), #RandomSplitter(valid_pct=0.15, seed=42)
                                     get_y=parent_label,
                                     batch_tfms=None)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("The model will be running on", device, "device")


# Creating train dataloader.
train_dls = jorda1_footbed_stitching.dataloaders(path, num_workers=4, batch_size=bs, device=device)

# Adding for gpu support.
train_dls = train_dls.cuda()

def load_model_weights(model, model_path):
    state = torch.load(model_path, map_location='cpu')
    for key in model.state_dict():
        if 'num_batches_tracked' in key:
            continue
        p = model.state_dict()[key]
        if key in state['state_dict']:
            ip = state['state_dict'][key]
            if p.shape == ip.shape:
                p.data.copy_(ip.data)  # Copy the data of parameters
            else:
                print(
                    'could not load layer: {}, mismatch shape {} ,{}'.format(key, (p.shape), (ip.shape)))
        else:
            print('could not load layer: {}, not in checkpoint'.format(key))
    return model

def create_timm_body(cut=None, n_in=3):
    
    model = timm.create_model('resnet50', pretrained=False, num_classes=11221)
    model = load_model_weights(model, '/tmp/resnet50_miil_21k.pth')
    _update_first_layer(model, n_in, pretrained=True)
    
    if cut is None:
        ll = list(enumerate(model.children()))
        cut = next(i for i,o in reversed(ll) if has_pool_type(o))
        
    if isinstance(cut, int):
        return nn.Sequential(*list(model.children())[:cut])
    elif callable(cut): 
        return cut(model)
    else: 
        raise NamedError("cut must be either integer or function")

def create_timm_model(n_out, cut=None, pretrained=True, n_in=3, init=nn.init.kaiming_normal_, custom_head=None,
                     concat_pool=True, **kwargs):
    
    body = create_timm_body()
    
    if custom_head is None:
        nf = num_features_model(nn.Sequential(*body.children()))
        head = create_head(nf, n_out, concat_pool=concat_pool, **kwargs)
    else:
        head = custom_head
        
    model = nn.Sequential(body, head)
    
    if init is not None:
        apply_init(model[1], init)
        
    return model

def timm_learner(dls, loss_func=None, pretrained=True, cut=None, splitter=None,
                y_range=None, config=None, n_out=None, normalize=True, **kwargs):
    
    if config is None: 
        config = {}
        
    if n_out is None: 
        n_out = get_c(dls)
        
    assert n_out, "`n_out` is not defined, and could not be inferred from data, set `dls.c` or pass `n_out`"
    
    if y_range is None and 'y_range' in config:
        y_range = config.pop('y_range')
        
    model = create_timm_model(n_out, default_split, pretrained, y_range=y_range, **config)
    learn = Learner(dls, model, loss_func=loss_func, splitter=default_split, **kwargs)
    
    if pretrained:
        learn.freeze()
        
    return learn

model_path = Path('/dbfs/tmp/vaibhav')

learn = timm_learner(train_dls, path=model_path, metrics=[error_rate, accuracy, Precision(), Recall()])

#learn = timm_learner(train_dls, 'convnext_base_in22k', metrics=[error_rate, accuracy, Precision(), Recall()]).to_fp16()

learn.model = learn.model.cuda()

print(learn.summary())

learn.fine_tune(10, base_lr=2e-4, freeze_epochs=5, cbs=[SaveModelCallback(fname='resnet50_miil_21k', every_epoch=True, monitor='accuracy')])
