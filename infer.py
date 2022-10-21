from ast import arg
import os
import sys
import cv2
import glob
import torch
import numpy as np
from torch import nn as nn
# from arch.gfpganv1_clean_arch import GFPGANv1Clean
from arch.gfpganv1_clean_arch_constant import GFPGANv1Clean
from basicsr.utils import img2tensor, tensor2img
from basicsr.utils import imwrite
from torchvision.transforms.functional import normalize

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
def enhance(model, img):
    restored_faces = []
    cropped_faces = []
    # the inputs are already aligned
    img = cv2.resize(img, (512, 512))
    cropped_faces = [img]

    # face restoration
    for cropped_face in cropped_faces:
        # prepare data
        cropped_face_t = img2tensor(cropped_face / 255., bgr2rgb=True, float32=True)
        normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
        cropped_face_t = cropped_face_t.unsqueeze(0).to(device)
        # convert to image
        # output = model(cropped_face_t, return_rgb=False)[0]
        # restored_face = tensor2img(output.squeeze(0), rgb2bgr=True, min_max=(-1, 1))
        try:
            output = model(cropped_face_t, return_rgb=False)[0]
            # convert to image
            restored_face = tensor2img(output.squeeze(0), rgb2bgr=True, min_max=(-1, 1))
        except RuntimeError as error:
            print(f'\tFailed inference for GFPGAN: {error}.')
            restored_face = cropped_face
        restored_face = restored_face.astype('uint8')
        restored_faces.append(restored_face)
    return cropped_faces, restored_faces, None

def infer(model, img_input):
    #model = model.to(device)
    #----------------------------------------args-----------------------------
    output = 'results/'
    suffix = None
    ext = 'auto'

    if img_input.endswith('/'):
        img_input = img_input[:-1]
    if os.path.isfile(img_input):
        img_list = [img_input]
    else:
        img_list = sorted(glob.glob(os.path.join(img_input, '*')))
    os.makedirs(output, exist_ok=True)
    print()
#------------------------ restore ------------------------
    for img_path in img_list:
        # read image
        img_name = os.path.basename(img_path)
        print(f'Processing {img_name} ...')
        basename, ext = os.path.splitext(img_name)
        input_img = cv2.imread(img_path, cv2.IMREAD_COLOR)

        # restore faces and background if necessary
        cropped_faces, restored_faces, restored_img = enhance(model,input_img)

        # save faces
        for idx, (cropped_face, restored_face) in enumerate(zip(cropped_faces, restored_faces)):
            # save cropped face
            # save_crop_path = os.path.join(output, 'cropped_faces', f'{basename}_{idx:02d}.png')
            # imwrite(cropped_face, save_crop_path)
            # save restored face
            if suffix is not None:
                save_face_name = f'{basename}_{idx:02d}_{suffix}.png'
            else:
                save_face_name = f'{basename}_{idx:02d}.png'
            save_restore_path = os.path.join(output, 'restored_faces', save_face_name)
            imwrite(restored_face, save_restore_path)
            # save comparison image
            cmp_img = np.concatenate((cropped_face, restored_face), axis=1)
            imwrite(cmp_img, os.path.join(output, 'cmp', f'{basename}_{idx:02d}.png'))

        # save restored img
        if restored_img is not None:
            if ext == 'auto':
                extension = ext[1:]
            else:
                extension = ext

            if suffix is not None:
                save_restore_path = os.path.join(output, 'restored_imgs', f'{basename}_{suffix}.{extension}')
            else:
                save_restore_path = os.path.join(output, 'restored_imgs', f'{basename}.{extension}')
            imwrite(restored_img, save_restore_path)

    print(f'Results are in the [{output}] folder.')
    return

def model(weight = 'models/GFPGANv1.3.pth'):
    model = GFPGANv1Clean(
                out_size=512,
                num_style_feat=512,
                channel_multiplier=2,
                num_mlp=8,
                input_is_latent=True,
                different_w=True,
                narrow=1,
                sft_half=True)
    if weight :
        state_dict = torch.load(weight,map_location=device)
        if 'params_ema' in state_dict:
            keyname = 'params_ema'
        else:
            keyname = 'params'
        model.load_state_dict(state_dict[keyname],strict=False)
    return model

if __name__ == '__main__':
    mod = model()
    infer(mod, img_input = 'input_img/cropped_img')
    # print(mod)