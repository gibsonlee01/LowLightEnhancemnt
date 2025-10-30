import os
import os.path as osp

import torch
import torch.utils.data as data

import numpy as np
from PIL import Image
import glob
import random
from torchvision.transforms import Compose, ToTensor, Normalize, ConvertImageDtype

random.seed(1143)

def populate_train_list(images_path, mode='train'):
    # print(images_path)
    image_list_lowlight = glob.glob(images_path + '*.png')
    train_list = image_list_lowlight
    if mode == 'train':
        random.shuffle(train_list)

    return train_list

class lowlight_loader(data.Dataset):

    def __init__(self, images_path, mode='test', normalize=True, resample='bicubic'):
        self.train_list = populate_train_list(images_path, mode)
        #self.h, self.w = int(img_size[0]), int(img_size[1])
        # train or test
        self.mode = mode
        self.data_list = self.train_list
        self.normalize = normalize
    # Data Augmentation
    # TODO: more data augmentation methods
    def FLIP_LR(self, low, high):
        if random.random() > 0.5:
            low = low.transpose(Image.FLIP_LEFT_RIGHT)
            high = high.transpose(Image.FLIP_LEFT_RIGHT)
        return low, high

    def FLIP_UD(self, low, high):
        if random.random() > 0.5:
            low = low.transpose(Image.FLIP_TOP_BOTTOM)
            high = high.transpose(Image.FLIP_TOP_BOTTOM)
        return low, high
    
    def get_params(self, low):
        self.w, self.h = low.size
        
        self.crop_height = random.randint(self.h//2, self.h) #random.randint(self.MinCropHeight, self.MaxCropHeight)
        self.crop_width = random.randint(self.w//2, self.w) #random.randint(self.MinCropWidth,self.MaxCropWidth)
        # self.crop_height = 224 #random.randint(self.MinCropHeight, self.MaxCropHeight)
        # self.crop_width = 224 #random.randint(self.MinCropWidth,self.MaxCropWidth)

        i = random.randint(0, self.h - self.crop_height)
        j = random.randint(0, self.w - self.crop_width)
        return i,j

    def Random_Crop(self, low, high):
        self.i,self.j = self.get_params((low))
        if random.random() > 0.5:
            low = low.crop((self.j, self.i, self.j + self.crop_width, self.i + self.crop_height))
            high = high.crop((self.j, self.i, self.j + self.crop_width, self.i + self.crop_height))
        return low, high
    
    
    def __getitem__(self, index):
        data_lowlight_path = self.data_list[index]
        
        if self.mode == 'train':
            data_lowlight = Image.open(data_lowlight_path)
            data_highlight = Image.open(data_lowlight_path.replace('low', 'normal').replace('Low','Normal'))
            
            data_lowlight, data_highlight = self.FLIP_LR(data_lowlight, data_highlight)
            data_lowlight, data_highlight = self.FLIP_UD(data_lowlight, data_highlight)
            data_lowlight, data_highlight = self.Random_Crop(data_lowlight, data_highlight)
            
            # print(self.w, self.h)
            #print(data_lowlight.size, data_highlight.size)
            
            data_lowlight = data_lowlight.resize((self.w, self.h), Image.Resampling.LANCZOS)
            data_highlight = data_highlight.resize((self.w, self.h), Image.Resampling.LANCZOS)
            data_lowlight, data_highlight = (np.asarray(data_lowlight) / 255.0), (np.asarray(data_highlight) / 255.0)

            if self.normalize:
                #data_lowlight, data_highlight = torch.from_numpy(data_lowlight).float(), torch.from_numpy(data_highlight).float()
                transform_input = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ConvertImageDtype(torch.float), ])
                transform_gt = Compose([ToTensor(), ConvertImageDtype(torch.float), ])
                #return transform_input(data_lowlight).permute(2, 0, 1), transform_gt(data_highlight).permute(2, 0, 1)
                return transform_input(data_lowlight), transform_gt(data_highlight)
            else:
                data_lowlight, data_highlight = torch.from_numpy(data_lowlight).float(), torch.from_numpy(data_highlight).float()
                return data_lowlight.permute(2,0,1), data_highlight.permute(2,0,1)

        elif self.mode == 'test':
            data_lowlight = Image.open(data_lowlight_path)
            data_highlight = Image.open(data_lowlight_path.replace('low', 'normal').replace('Low','Normal'))
            data_lowlight, data_highlight = (np.asarray(data_lowlight) / 255.0), (np.asarray(data_highlight) / 255.0)
            #data_lowlight, data_highlight = torch.from_numpy(data_lowlight).float(), torch.from_numpy(data_highlight).float()
            if self.normalize:
                #data_lowlight, data_highlight = torch.from_numpy(data_lowlight).float(), torch.from_numpy(data_highlight).float()
                transform_input = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ConvertImageDtype(torch.float), ])
                transform_gt = Compose([ToTensor(), ConvertImageDtype(torch.float), ])
                #return transform_input(data_lowlight).permute(2, 0, 1), transform_gt(data_highlight).permute(2, 0, 1)
                return transform_input(data_lowlight), transform_gt(data_highlight)
            else:
                data_lowlight, data_highlight = torch.from_numpy(data_lowlight).float(), torch.from_numpy(data_highlight).float()
                return data_lowlight.permute(2,0,1), data_highlight.permute(2,0,1)
            
    def __len__(self):
        return len(self.data_list)


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    images_path = '/data/unagi0/cui_data/light_dataset/LOL_v2/Train/Low/'

    train_dataset = lowlight_loader(images_path)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=4,
                                               pin_memory=True)
    for iteration, imgs in enumerate(train_loader):
        print(iteration)
        print(imgs[0].shape)
        print(imgs[1].shape)
        
#============ Env_dataloader==================

class lowlight_loader_env(data.Dataset):

    def __init__(self, images_path, mode='train', normalize=True, envmap_root=None):
        self.train_list = populate_train_list(images_path, mode)
        self.mode = mode
        self.data_list = self.train_list
        self.normalize = normalize
        self.envmap_root = envmap_root  # npy 저장된 경로        

    # Data Augmentation
    # TODO: more data augmentation methods
    def FLIP_LR(self, low, high):
        if random.random() > 0.5:
            low = low.transpose(Image.FLIP_LEFT_RIGHT)
            high = high.transpose(Image.FLIP_LEFT_RIGHT)
        return low, high

    def FLIP_UD(self, low, high):
        if random.random() > 0.5:
            low = low.transpose(Image.FLIP_TOP_BOTTOM)
            high = high.transpose(Image.FLIP_TOP_BOTTOM)
        return low, high
    
    def get_params(self, low):
        self.w, self.h = low.size
        
        self.crop_height = random.randint(self.h//2, self.h) #random.randint(self.MinCropHeight, self.MaxCropHeight)
        self.crop_width = random.randint(self.w//2, self.w) #random.randint(self.MinCropWidth,self.MaxCropWidth)
        # self.crop_height = 224 #random.randint(self.MinCropHeight, self.MaxCropHeight)
        # self.crop_width = 224 #random.randint(self.MinCropWidth,self.MaxCropWidth)

        i = random.randint(0, self.h - self.crop_height)
        j = random.randint(0, self.w - self.crop_width)
        return i,j

    def Random_Crop(self, low, high):
        self.i,self.j = self.get_params((low))
        if random.random() > 0.5:
            low = low.crop((self.j, self.i, self.j + self.crop_width, self.i + self.crop_height))
            high = high.crop((self.j, self.i, self.j + self.crop_width, self.i + self.crop_height))
        return low, high

    def __getitem__(self, index):
        data_lowlight_path = self.data_list[index]
        data_highlight_path = data_lowlight_path.replace('low', 'normal').replace('Low','Normal')

        # ====== load lowlight / highlight ======
        data_lowlight = Image.open(data_lowlight_path)
        data_highlight = Image.open(data_highlight_path)

        if self.mode == 'train':
            data_lowlight, data_highlight = self.FLIP_LR(data_lowlight, data_highlight)
            data_lowlight, data_highlight = self.FLIP_UD(data_lowlight, data_highlight)
            data_lowlight, data_highlight = self.Random_Crop(data_lowlight, data_highlight)

            data_lowlight = data_lowlight.resize((self.w, self.h), Image.Resampling.LANCZOS)
            data_highlight = data_highlight.resize((self.w, self.h), Image.Resampling.LANCZOS)
            data_lowlight = np.asarray(data_lowlight) / 255.0
            data_highlight = np.asarray(data_highlight) / 255.0

        if self.normalize:
            transform_input = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ConvertImageDtype(torch.float)])
            transform_gt = Compose([ToTensor(), ConvertImageDtype(torch.float)])
            low_tensor = transform_input(data_lowlight)
            high_tensor = transform_gt(data_highlight)
        else:
            low_tensor = torch.from_numpy(data_lowlight).float().permute(2,0,1)
            high_tensor = torch.from_numpy(data_highlight).float().permute(2,0,1)

        # ====== train일 때만 env_map 로드 ======
        if self.mode == "train" and self.envmap_root is not None:
            base_name = os.path.basename(data_highlight_path)  # e.g. normal00001.png
            base_name = base_name.replace('.png', '_env_map.npy')  # normal00001_env_map.npy
            env_path = os.path.join(self.envmap_root, base_name)
            env_map = np.load(env_path)   # (H,W,3)
            env_map = torch.from_numpy(env_map).float().permute(2,0,1)  # (3,H,W)
            return low_tensor, high_tensor, env_map

        # ====== test 모드에서는 env_map 없음 ======
        return low_tensor, high_tensor

    def __len__(self):
        return len(self.data_list)


# ========== SH Coefficients Utility ==========
def compute_sh_coefficients(env_map, sh_order=2):
    """
    env_map: (H, W, 3) numpy array, HDR environment map
    return: ( (sh_order+1)^2 * 3, ) torch.FloatTensor
            ex) sh_order=2 -> (27,)
    """
    H, W, C = env_map.shape
    assert C == 3, f"Env map channel mismatch: {C}"

    num_coeffs = (sh_order + 1) ** 2  # 9 for order=2

    # Spherical coordinates
    theta = np.linspace(0, np.pi, H)[:, None]
    phi = np.linspace(0, 2*np.pi, W)[None, :]
    theta, phi = np.meshgrid(theta, phi, indexing="ij")

    dOmega = np.sin(theta)

    # Basis functions
    from scipy.special import sph_harm
    Y = []
    for l in range(sh_order+1):
        for m in range(-l, l+1):
            Ylm = sph_harm(m, l, phi, theta).real  # (H, W)
            Y.append(Ylm)
    Y = np.stack(Y, axis=-1)  # (H, W, num_coeffs)

    # Compute coefficients
    coeffs = []
    for c in range(3):
        f = env_map[..., c]
        clm = np.sum(f[..., None] * Y * dOmega[..., None],
                     axis=(0, 1)) * (np.pi/H) * (2*np.pi/W)
        coeffs.append(clm)  # (num_coeffs,)
    coeffs = np.stack(coeffs, axis=0)   # (3, num_coeffs)
    coeffs = coeffs.T.reshape(-1)       # (num_coeffs*3,)

    coeffs = torch.from_numpy(coeffs).float()
    assert coeffs.shape[0] == num_coeffs * 3, \
        f"SH coeff shape mismatch: {coeffs.shape}"
    return coeffs


# ========== Dataset ==========

class lowlight_loader_sh(data.Dataset):
    def __init__(self, images_path, mode='train', normalize=True,
                 envmap_root=None, sh_order=2):
        from data_loaders.lol import populate_train_list
        self.train_list = populate_train_list(images_path, mode)
        self.mode = mode
        self.data_list = self.train_list
        self.normalize = normalize
        self.envmap_root = envmap_root
        self.sh_order = sh_order

    def FLIP_LR(self, low, high):
        if random.random() > 0.5:
            low = low.transpose(Image.FLIP_LEFT_RIGHT)
            high = high.transpose(Image.FLIP_LEFT_RIGHT)
        return low, high

    def FLIP_UD(self, low, high):
        if random.random() > 0.5:
            low = low.transpose(Image.FLIP_TOP_BOTTOM)
            high = high.transpose(Image.FLIP_TOP_BOTTOM)
        return low, high

    def get_params(self, low):
        self.w, self.h = low.size
        self.crop_height = random.randint(self.h//2, self.h)
        self.crop_width = random.randint(self.w//2, self.w)
        i = random.randint(0, self.h - self.crop_height)
        j = random.randint(0, self.w - self.crop_width)
        return i, j

    def Random_Crop(self, low, high):
        self.i, self.j = self.get_params(low)
        if random.random() > 0.5:
            low = low.crop((self.j, self.i, self.j+self.crop_width, self.i+self.crop_height))
            high = high.crop((self.j, self.i, self.j+self.crop_width, self.i+self.crop_height))
        return low, high

    def __getitem__(self, index):
        data_lowlight_path = self.data_list[index]
        data_highlight_path = data_lowlight_path.replace('low', 'normal').replace('Low', 'Normal')

        data_lowlight = Image.open(data_lowlight_path)
        data_highlight = Image.open(data_highlight_path)

        if self.mode == 'train':
            data_lowlight, data_highlight = self.FLIP_LR(data_lowlight, data_highlight)
            data_lowlight, data_highlight = self.FLIP_UD(data_lowlight, data_highlight)
            data_lowlight, data_highlight = self.Random_Crop(data_lowlight, data_highlight)

            data_lowlight = data_lowlight.resize((self.w, self.h), Image.Resampling.LANCZOS)
            data_highlight = data_highlight.resize((self.w, self.h), Image.Resampling.LANCZOS)
            data_lowlight = np.asarray(data_lowlight) / 255.0
            data_highlight = np.asarray(data_highlight) / 255.0

        if self.normalize:
            transform_input = Compose([
                ToTensor(),
                Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ConvertImageDtype(torch.float)
            ])
            transform_gt = Compose([
                ToTensor(),
                ConvertImageDtype(torch.float)
            ])
            low_tensor = transform_input(data_lowlight)
            high_tensor = transform_gt(data_highlight)
        else:
            low_tensor = torch.from_numpy(data_lowlight).float().permute(2, 0, 1)
            high_tensor = torch.from_numpy(data_highlight).float().permute(2, 0, 1)

        if self.mode == "train" and self.envmap_root is not None:
            base_name = os.path.basename(data_highlight_path).replace('.png', '_env_map.npy')
            env_path = os.path.join(self.envmap_root, base_name)
            env_map = np.load(env_path)
            sh_coeffs = compute_sh_coefficients(env_map, sh_order=self.sh_order)
            return low_tensor, high_tensor, sh_coeffs

        return low_tensor, high_tensor

    def __len__(self):
        return len(self.data_list)
