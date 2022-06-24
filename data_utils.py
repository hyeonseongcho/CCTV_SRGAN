from os import listdir
from os.path import join

from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])


def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)


def train_hr_transform(crop_size):
    return Compose([
        RandomCrop(crop_size),
        ToTensor(),
    ])


def train_lr_transform(crop_size, upscale_factor):
    return Compose([
        ToPILImage(),
        Resize(crop_size // upscale_factor, interpolation=Image.BICUBIC),
        ToTensor()
    ])


def display_transform():
    return Compose([
        ToPILImage(),
        Resize(400),
        CenterCrop(400),
        ToTensor()
    ])

class CustumDataset(torch.utils.data.Dataset):
    def __init__(self, img_root, high_res, upscale_factor):
        self.img_root = img_root
        self.paths = list(Path(self.img_root).glob('*'))

        self.hr_transform = torchvision.transforms.Compose([
                            torchvision.transforms.Resize(2 * high_res),
                            torchvision.transforms.RandomCrop((high_res, high_res)),
                            torchvision.transforms.ToTensor(),
                            ])

        self.lr_transform = torchvision.transforms.Compose([
                            torchvision.transforms.ToPILImage(),
                            torchvision.transforms.Resize(high_res // upscale_factor, interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
                            torchvision.transforms.ToTensor(),
                            ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(str(path)).convert('RGB')
        
        target = self.hr_transform(img)
        data = self.lr_transform(target)

        return data, target