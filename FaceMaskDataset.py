import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
from glob import glob
from os.path import basename


DATA_TRANSFORMS = {
    'train' : transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), 
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]), 
    'eval' : transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}


class FaceMaskDataset(Dataset):

    def __init__(self,
                 root_dir,
                 have_label,
                 phase='train'):
        """
        Args:
            root_dir (string): Directory with all the images.
            have_label (boolean): Flag for images having the labels in their names.
            phase (string): test or eval (For transformations).
        """
        self.root_dir = root_dir
        self.have_label = have_label
        self.transform = DATA_TRANSFORMS[phase]
        self.images_paths = glob(f'{self.root_dir}/*.jpg')
        self.labels = None
        if self.have_label:
            self.labels = torch.LongTensor([int(image_path.split('.jpg')[0][-1]) for image_path in self.images_paths])

    def __len__(self):
        return len(self.images_paths)

    def __getitem__(self, idx):
        image_path = self.images_paths[idx]
        image = Image.open(image_path)
        image = (basename(image_path), self.transform(image))
        if self.have_label:
            label = self.labels[idx]
            item = (*image, label)
        else:
            item = image
        return item