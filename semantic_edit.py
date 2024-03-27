import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
import torch
from sema_guidance import StableDiffusionFreeGuidancePipeline
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from tqdm.auto import tqdm
from torch.nn import init
from utils.guidance_functions import *
import argparse
from diffusers import LMSDiscreteScheduler, DDPMScheduler, DDIMScheduler, DPMSolverMultistepScheduler
from utils import *
from PIL import Image

torch.cuda.manual_seed_all(1234) 
torch.set_printoptions(precision=2, linewidth=140, sci_mode=False)
mpl.rcParams['image.cmap'] = 'gray_r'

# Load model
print("Start Inference!")
model_id = "runwayml/stable-diffusion-v1-5"
device = "cuda"
NUM_DDIM_STEPS = 50
pipe = StableDiffusionFreeGuidancePipeline.from_pretrained(model_id, torch_dtype=torch.float16, variant="fp16")
pipe.unet = UNetDistributedDataParallel(pipe.unet, device_ids=[0]).cuda()
pipe.unet = pipe.unet.module
pipe = pipe.to(device)

pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config) # DDPMScheduler -> DDIMScheduler
pipe.scheduler.set_timesteps(NUM_DDIM_STEPS)

pipe.enable_attention_slicing()
torch.backends.cudnn.benchmark = True

seed = 40289 #int(torch.rand((1,)) * 100000) #13519
generator=torch.manual_seed(seed)
print(seed)

def get_data(path,dataset_name,index=2):
    if dataset_name not in ['ImageNet','Imagen','Editing-Mask']:
        print('error dataset name!')
        return None
    data_pair=[]
    with open(path,'r',encoding='utf-8') as f:
        for i in f.readlines():
            i=i.strip('\n')
            l=i.split(' <> ')
            if dataset_name=='ImageNet':
                source = l[-2].split(' ')[-1]
                target = l[-1].split(' ')[-1]
                data_pair.append([l[0]]+['a photo of a '+source]+['a photo of a '+target]+[target])
            elif dataset_name=='Imagen':
                data_pair.append([l[0]]+[l[1]+'_'+str(index)+'.jpg']+[l[-2]]+[l[-1]])
            elif dataset_name=='Editing-Mask':
                if len(l)==4:
                    data_pair.append([l[0]]+[l[1]+'_'+str(index)+'.jpg']+[l[-2]]+[l[-1]])
                else:
                    data_pair.append([l[0][18:-5]]+l[:2]+['a '+l[-1]])
        print("total numbers is : "+str(len(data_pair)))
    return data_pair

def main():
    data_pair = get_data("./dataset_txt/ImageNet_list.txt", "ImageNet")
    output_path = "./results/"
    os.makedirs(output_path, exist_ok=True)
    
    for data in data_pair:
        try:
            # setting
            ids = data[0][18:-5]
            source_prompt = data[1]
            target_prompt = data[2]
            
            image_path = './dataset/ImageNet/' + data[0]
            input_image = Image.open(image_path).resize((512,512)).convert("RGB")
            input_image.save(output_path + ids + '-or_image.jpg')
            
            # editing set        
            prompts = [source_prompt,
                    target_prompt]
            
            object_to_edit = data[3]

            guidance = partial(match_semantic_feature, position_weight=6.0, sem_weight=3.0, feature_weight=6.0) #feature_weight: 3.0
            feature_layer = pipe.unet.up_blocks[-3].resnets[-3] #resnets: 0 1 2

            latents = get_ddim_latents(pipe, image_path, prompts[0], device)
            init_latent = latents[-1]

            image_list = pipe(prompts[1], prompts[0], obj_to_edit = object_to_edit, height=512, width=512, 
                            num_inference_steps=50, generator=generator, latents=init_latent, all_latents=latents,
                            max_guidance_iter_per_step=15, guidance_func=guidance, g_weight=1500, feature_layer=feature_layer) 
            
            image_list[0][0].save(output_path + ids + '_edit.jpg')
            image_list[1][0].save(output_path + ids + '_recon.jpg')
            image_list[2].save(output_path + ids + '_mask.jpg')

        except:
            continue

if __name__=="__main__":
    main()