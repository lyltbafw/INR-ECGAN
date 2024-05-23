import os.path
from data.base_dataset import BaseDataset, get_params, get_transform, normalize
from data.image_folder import make_dataset
from PIL import Image
from .coord import to_pixel_samples
import random
import torch
class AlignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot    

        dir_A = '_A'
        self.dir_A = os.path.join(opt.dataroot, opt.phase + dir_A)
        self.A_paths = sorted(make_dataset(self.dir_A))

        if True:
            dir_B = '_B'
            self.dir_B = os.path.join(opt.dataroot, opt.phase + dir_B)  
            self.B_paths = sorted(make_dataset(self.dir_B))


        self.dataset_size = len(self.A_paths) 
      
    def __getitem__(self, index):
        A_path = self.A_paths[index]              
        A = Image.open(A_path)        
        params = get_params(self.opt, A.size)
        transform_A = get_transform(self.opt, params, grayscale=(self.opt.input_nc == 1))
        A_tensor = transform_A(A)


        B_tensor = inst_tensor = feat_tensor = 0
        if True:
            B_path = self.B_paths[index]   
            B = Image.open(B_path).convert('RGB')
            transform_B = get_transform(self.opt, params)      
            B_tensor = transform_B(B)


        if self.opt.netG == 'INRs':
            if True:
                hflip = random.random() < 0.5
                vflip = random.random() < 0.5
                dflip = random.random() < 0.5
                # print(A_tensor)
                def augment(x):
                    # print(x)
                    if hflip:
                        x = x.flip(-2)
                    if vflip:
                        x = x.flip(-1)
                    if dflip:
                        x = x.transpose(-2, -1)
                    return x

                crop_sar = augment(A_tensor)
                crop_rgb = augment(B_tensor)
            coord, coord_rgb = to_pixel_samples(crop_rgb.contiguous())


            cell = torch.ones_like(coord)
            cell[:, 0] *= 2 / crop_rgb.shape[-2]
            cell[:, 1] *= 2 / crop_rgb.shape[-1]

            input_dict = {'label': A_tensor,  'image': B_tensor,
                          'feat': feat_tensor, 'path': A_path, 'coord': coord, 'gt_rgb': coord_rgb, 'cell': cell}
        else:
            input_dict = {'label': A_tensor,  'image': B_tensor,
                      'feat': feat_tensor, 'path': A_path}

        return input_dict

    def __len__(self):
        return len(self.A_paths) // self.opt.batchSize * self.opt.batchSize

    def name(self):
        return 'AlignedDataset'
