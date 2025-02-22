import os 
import pathlib
import torch
import torch.nn.functional as F
from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset
import cv2 

def extract_JSRT_segs(path):
    segs_files = []

    for dirpath, _, filenames in os.walk(path):
        for file in filenames:
            full_path = os.path.join(dirpath, file)
            segs_files.append(full_path)
    return segs_files

def get_jsrt_datasets(transforms=None):
    data_path = pathlib.Path('datasets/JSRT')
    seg_img_path = os.path.join(data_path, 'seg_images')
    ref_img_path = os.path.join(data_path, 'reference_images')
    label_path = os.path.join(data_path, 'landmarks')
    segs_path = os.path.join(data_path, 'DifferentQualities')

    ref_images = os.listdir(ref_img_path)
    segs = extract_JSRT_segs(segs_path)    
    
    d_eval = LandmarksDataset(images=segs, img_path=seg_img_path, label_path=label_path, is_seg=True, organ=['L', 'H'], transform=transforms) 
    d_reference = LandmarksDataset(images=ref_images, img_path=ref_img_path, label_path=label_path, is_seg=False, organ=['L', 'H'], transform=transforms) 

    return d_reference, d_eval

class LandmarksDataset(Dataset):
    def __init__(self, images, img_path, label_path, is_seg=False, transform=None, organ = False):
        
        self.images = images 
        self.img_path = img_path
        self.label_path = label_path
        
        self.RL_path = os.path.join(self.label_path, 'RL')
        self.LL_path = os.path.join(self.label_path, 'LL')        
        self.H_path = os.path.join(self.label_path, 'H')
        self.RCLA_path = os.path.join(self.label_path, 'RCLA')
        self.LCLA_path = os.path.join(self.label_path, 'LCLA')
            
        self.organ = organ
        self.is_seg = is_seg
        self.transform = transform

    def __len__(self):
        return len(self.images)
        
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.basename(self.images[idx])
        
        img_path = os.path.join(self.img_path, img_name)
                
        RL_path = os.path.join(self.RL_path, img_name.replace('.png', '.npy'))       
        LL_path = os.path.join(self.LL_path, img_name.replace('.png', '.npy'))
        H_path = os.path.join(self.H_path, img_name.replace('.png', '.npy'))
        RCLA_path = os.path.join(self.RCLA_path, img_name.replace('.png', '.npy'))       
        LCLA_path = os.path.join(self.LCLA_path, img_name.replace('.png', '.npy'))
                           
        image = io.imread(img_path).astype('float') / 255.0
        image = np.expand_dims(image, axis=2)

        organs = []
        
        if 'L' in self.organ:    
            RL = np.load(RL_path).astype('float')
            LL = np.load(LL_path).astype('float')
            organs.append(RL)
            organs.append(LL)
        
        try:
            if 'H' in self.organ: 
                H = np.load(H_path).astype('float')
                organs.append(H)
        except:
            pass

        try:
            if 'C' in self.organ:
                RCLA = np.load(RCLA_path).astype('float')
                LCLA = np.load(LCLA_path).astype('float')
                organs.append(RCLA)
                organs.append(LCLA)
        except:
            pass

        landmarks = np.concatenate(organs, axis = 0)
                
        sample = {'image': image, 'landmarks': landmarks}
        
        if self.is_seg:
            seg = io.imread(self.images[idx]).astype('int')
            sample['seg'] = seg

        if self.transform:
            sample = self.transform(sample)

        return sample


class Rescale(object):
    def __init__(self, output_size=1024):
        """
        Args:
            output_size (int): Desired output size (same for both height and width).
        """
        self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        
        h, w = image.shape[:2]
        
        # Rescale both height and width to the given output size
        scale_factor = self.output_size / h  # Same for both h and w
        new_h, new_w = self.output_size, self.output_size

        img = transform.resize(image, (new_h, new_w), preserve_range=True)

        landmarks[:, 0] = landmarks[:, 0] * new_w / w
        landmarks[:, 1] = landmarks[:, 1] * new_h / h

        if 'seg' in sample:
            seg = transform.resize(sample['seg'], (new_h, new_w), preserve_range=True, anti_aliasing=False, order=0)
            sample['seg'] = seg
        
        sample['image'] = img
        sample['landmarks'] = landmarks
        
        return sample


class RandomScale(object):
    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']       
        
        # Pongo limites para evitar que los landmarks salgan del contorno
        min_x = np.min(landmarks[:,0]) 
        max_x = np.max(landmarks[:,0])
        ancho = max_x - min_x
        
        min_y = np.min(landmarks[:,1])
        max_y = np.max(landmarks[:,1])
        alto = max_y - min_y
        
        max_var_x = 1024 / ancho 
        max_var_y = 1024 / alto
                
        min_var_x = 0.80
        min_var_y = 0.80
                                
        varx = np.random.uniform(min_var_x, max_var_x)
        vary = np.random.uniform(min_var_x, max_var_y)
                
        landmarks[:,0] = landmarks[:,0] * varx
        landmarks[:,1] = landmarks[:,1] * vary
        
        h, w = image.shape[:2]
        new_h = np.round(h * vary).astype('int')
        new_w = np.round(w * varx).astype('int')

        img = transform.resize(image, (new_h, new_w))
        
        # Cropeo o padeo aleatoriamente
        min_x = np.round(np.min(landmarks[:,0])).astype('int')
        max_x = np.round(np.max(landmarks[:,0])).astype('int')
        
        min_y = np.round(np.min(landmarks[:,1])).astype('int')
        max_y = np.round(np.max(landmarks[:,1])).astype('int')
        
        if new_h > 1024:
            rango = 1024 - (max_y - min_y)
            maxl0y = new_h - 1025
            
            if rango > 0 and min_y > 0:
                l0y = min_y - np.random.randint(0, min(rango, min_y))
                l0y = min(maxl0y, l0y)
            else:
                l0y = min_y
                
            l1y = l0y + 1024
            
            img = img[l0y:l1y,:]
            landmarks[:,1] -= l0y
            
        elif new_h < 1024:
            pad = h - new_h
            p0 = np.random.randint(np.floor(pad/4), np.ceil(3*pad/4))
            p1 = pad - p0
            
            img = np.pad(img, ((p0, p1), (0, 0), (0, 0)), mode='constant', constant_values=0)
            landmarks[:,1] += p0
        
        if new_w > 1024:
            rango = 1024 - (max_x - min_x)
            maxl0x = new_w - 1025
            
            if rango > 0 and min_x > 0:
                l0x = min_x - np.random.randint(0, min(rango, min_x))
                l0x = min(maxl0x, l0x)
            else:
                l0x = min_x
            
            l1x = l0x + 1024
                
            img = img[:, l0x:l1x]
            landmarks[:,0] -= l0x
            
        elif new_w < 1024:
            pad = w - new_w
            p0 = np.random.randint(np.floor(pad/4), np.ceil(3*pad/4))
            p1 = pad - p0
            
            img = np.pad(img, ((0, 0), (p0, p1), (0, 0)), mode='constant', constant_values=0)
            landmarks[:,0] += p0
        
        if img.shape[0] != 1024 or img.shape[1] != 1024:
            print('Original', [new_h,new_w])
            print('Salida', img.shape)
            raise Exception('Error')
            
        return {'image': img, 'landmarks': landmarks}


class FixedScale(object):
    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']       
        
        # Pongo limites para evitar que los landmarks salgan del contorno
        min_x = np.min(landmarks[:,0]) 
        max_x = np.max(landmarks[:,0])
        ancho = max_x - min_x
        
        min_y = np.min(landmarks[:,1])
        max_y = np.max(landmarks[:,1])
        alto = max_y - min_y
                        
        landmarks[:,0] = landmarks[:,0] * 600/ancho
        landmarks[:,1] = landmarks[:,1] * 750/alto
        
        h, w = image.shape[:2]
        new_h = np.round(h * 750/ancho).astype('int')
        new_w = np.round(w * 600/alto).astype('int')

        img = transform.resize(image, (new_h, new_w))
        
        # Cropeo o padeo aleatoriamente
        min_x = np.round(np.min(landmarks[:,0])).astype('int')
        max_x = np.round(np.max(landmarks[:,0])).astype('int')
        
        min_y = np.round(np.min(landmarks[:,1])).astype('int')
        max_y = np.round(np.max(landmarks[:,1])).astype('int')
        
        if new_h > 1024:
            rango = 1024 - (max_y - min_y)
            maxl0y = new_h - 1025
            
            if rango > 0 and min_y > 0:
                l0y = min_y - np.random.randint(0, min(rango, min_y))
                l0y = min(maxl0y, l0y)
            else:
                l0y = min_y
                
            l1y = l0y + 1024
            
            img = img[l0y:l1y,:]
            landmarks[:,1] -= l0y
            
        elif new_h < 1024:
            pad = h - new_h
            p0 = np.random.randint(np.floor(pad/4), np.ceil(3*pad/4))
            p1 = pad - p0
            
            img = np.pad(img, ((p0, p1), (0, 0), (0, 0)), mode='constant', constant_values=0)
            landmarks[:,1] += p0
        
        if new_w > 1024:
            rango = 1024 - (max_x - min_x)
            maxl0x = new_w - 1025
            
            if rango > 0 and min_x > 0:
                l0x = min_x - np.random.randint(0, min(rango, min_x))
                l0x = min(maxl0x, l0x)
            else:
                l0x = min_x
            
            l1x = l0x + 1024
                
            img = img[:, l0x:l1x]
            landmarks[:,0] -= l0x
            
        elif new_w < 1024:
            pad = w - new_w
            p0 = np.random.randint(np.floor(pad/4), np.ceil(3*pad/4))
            p1 = pad - p0
            
            img = np.pad(img, ((0, 0), (p0, p1), (0, 0)), mode='constant', constant_values=0)
            landmarks[:,0] += p0
        
        if img.shape[0] != 1024 or img.shape[1] != 1024:
            print('Original', [new_h,new_w])
            print('Salida', img.shape)
            raise Exception('Error')
            
        return {'image': img, 'landmarks': landmarks}


def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
 
    # apply gamma correction using the lookup table
    return np.float32(cv2.LUT(image.astype('uint8'), table))


class AugColor(object):
    def __init__(self, gammaFactor):
        self.gammaf = gammaFactor

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        
        # Gamma
        gamma = np.random.uniform(1 - self.gammaf, 1 + self.gammaf / 2)
        
        image[:,:,0] = adjust_gamma(image[:,:,0] * 255, gamma) / 255
        
        # Adds a little noise
        image = image + np.random.normal(0, 1/128, image.shape)
        
        return {'image': image, 'landmarks': landmarks}

    
class Rotate(object):
    def __init__(self, angle):
        self.angle = angle

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        
        angle = np.random.uniform(- self.angle, self.angle)

        image = transform.rotate(image, angle)
        
        centro = image.shape[0] / 2, image.shape[1] / 2
        
        landmarks -= centro
        
        theta = np.deg2rad(angle)
        c, s = np.cos(theta), np.sin(theta)
        R = np.array(((c, -s), (s, c)))
        
        landmarks = np.dot(landmarks, R)
        
        landmarks += centro

        return {'image': image, 'landmarks': landmarks}

    
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
                
        size = image.shape[0]
        image = image.transpose(2, 0, 1)
        landmarks = landmarks / size
        landmarks = np.clip(landmarks, 0, 1)
        
        return {'image': torch.from_numpy(image).float(),
                'landmarks': torch.from_numpy(landmarks).float()}


def getDenseMask(RL, LL, H = None, CLA1 = None, CLA2 = None, size=1024):
    img = np.zeros([size, size])
    
    RL = RL.reshape(-1, 1, 2).astype('int')
    LL = LL.reshape(-1, 1, 2).astype('int')

    img = cv2.drawContours(img, [RL], -1, 1, -1)
    img = cv2.drawContours(img, [LL], -1, 1, -1)
    
    if H is not None:
        H = H.reshape(-1, 1, 2).astype('int')
        img = cv2.drawContours(img, [H], -1, 2, -1)
        
    if CLA1 is not None:
        CLA1 = CLA1.reshape(-1, 1, 2).astype('int')
        img = cv2.drawContours(img, [CLA1], -1, 3, -1)
    
    if CLA2 is not None:
        CLA2 = CLA2.reshape(-1, 1, 2).astype('int')
        img = cv2.drawContours(img, [CLA2], -1, 3, -1)
    
    return img


class ToTensorSeg(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
                 
        size = image.shape[0]
        image = image.transpose((2, 0, 1))
        landmarks = landmarks / size
        landmarks = np.clip(landmarks, 0, 1) * size

        RL = landmarks[:44,:]
        LL = landmarks[44:94,:]

        if landmarks.shape[0] == 94:
            seg = getDenseMask(RL, LL, size=size)
        elif landmarks.shape[0] == 120:
            H = landmarks[94:120,:]
            seg = getDenseMask(RL, LL, H, size=size)
        else:
            H = landmarks[94:120,:]
            CLA1 = landmarks[120:143,:]
            CLA2 = landmarks[143:,:]
            seg = getDenseMask(RL, LL, H, CLA1, CLA2, size=size)
            
        sample['image'] = torch.from_numpy((image - image.min()) / (image.max() - image.min())).float() #torch.from_numpy(image).float()
        sample['GT'] = torch.from_numpy(seg).long().unsqueeze(0)
        del sample['landmarks'] # delete old key
        
        if 'seg' in sample:
            sample['seg'] = torch.from_numpy(sample['seg']).long().unsqueeze(0)
        
        return sample
    
class ToNumpy(object):
    """Convert tensors in sample to numpy arrays."""

    def __call__(self, sample):
        for key in sample.keys():
            if not torch.is_tensor(sample[key]):
                raise TypeError(f"{key} must be a tensor before conversion to numpy array.")
            
            array = sample[key].squeeze().numpy()

            if key != 'image':
                sample[key] = array.astype(np.uint8)
            else:
                img_np = (array * 255).astype(np.uint8)
                sample[key] = img_np
        return sample

class OneHot(object):
    """Convert tensor to one-hot version"""
    
    def __call__(self, sample):    
        for key in sample.keys():
            if key != 'image':
                mask = sample[key]
                
                if not torch.is_tensor(mask):
                    raise TypeError(f"{key} must be a tensor to convert to one-hot encoding.")

                n_classes = len(np.unique(sample[key].numpy())) if len(np.unique(sample[key].numpy())) > 2 else 1
                
                if n_classes == 1:
                    mask_onehot = mask / mask.max()
                else:
                    mask_onehot = F.one_hot(mask.squeeze().long(), num_classes=n_classes)
                    mask_onehot = mask_onehot.float().permute(2, 0, 1)
                
                sample[key] = mask_onehot

        return sample
    