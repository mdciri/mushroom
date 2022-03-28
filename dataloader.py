import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import VisionDataset
from pycocotools.coco import COCO

class LoadCocoDataset(VisionDataset):
    """ Load Coco Dataset class using pycocotools.
    Args:
        annFile (string): Path to json annotation file.
        transforms (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
    """

    def __init__(self, annFile, transforms=None):
        super(LoadCocoDataset).__init__()

        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.transforms = transforms
        self.num_classes = len(self.coco.cats.keys())

    def __len__(self):
        return len(self.ids)

    def _load_image(self, id):
        path = self.coco.loadImgs(id)[0]["file_name"]
        return Image.open(path).convert("RGB")

    def _load_target(self, id):
        return self.coco.loadAnns(self.coco.getAnnIds(imgIds=id))[0]["category_id"]

    def __getitem__(self, index):
        id = self.ids[index]
        image = self._load_image(id)
        target = self._load_target(id)

        if self.transforms is not None:
            image = self.transforms(image)

        target = F.one_hot(
            torch.tensor(target, dtype=torch.long),
            self.num_classes
            )
        target = torch.tensor(target, dtype=torch.float32)

        return image, target

    def get_category_info(self, index):
        id = self.ids[index]
        target = self._load_target(id)
        cat_name = self.coco.cats[target]["name"]
        supercat_name = self.coco.cats[target]["supercategory"]

        return cat_name, supercat_name

class LoadCocoTestDataset(VisionDataset):
    """ Load Coco Dataset class using pycocotools.
    Args:
        annFile (string): Path to json annotation file.
        transforms (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
    """

    def __init__(self, annFile, transforms=None):
        super(LoadCocoTestDataset).__init__()

        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.transforms = transforms

    def __len__(self):
        return len(self.ids)

    def _load_image(self, id):
        path = self.coco.loadImgs(id)[0]["file_name"]
        return Image.open(path).convert("RGB")

    def __getitem__(self, index):
        id = self.ids[index]
        image = self._load_image(id)

        if self.transforms is not None:
            image = self.transforms(image)

        return image, id