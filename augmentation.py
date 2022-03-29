import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF


def get_train_transforms(
    image_shape,
    perc_augmentation,
    perc_horiz_filp,
    perc_vert_filp,
    gamma_range,
    perc_bright,
    rotation_range,
    perc_rotation):

    transforms =  T.Compose([
        T.Resize(image_shape),
        T.RandomApply(
            [
                T.RandomHorizontalFlip(p=perc_horiz_filp),
                T.RandomVerticalFlip(p=perc_vert_filp),
                RandomAdjustGamma(gamma_range, p=perc_bright),
                RandomRotation(rotation_range, p=perc_rotation),
            ], p = perc_augmentation),  
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    return transforms

def get_test_val_transforms(image_shape):

    transforms =  T.Compose([
            T.Resize(image_shape),
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

    return transforms

class RandomAdjustGamma():
    """ Random gamma power brightness transform
    """

    def __init__(self, gamma_range, p=0.5):

        if isinstance(gamma_range, tuple):
            self.a, self.b = gamma_range
        if isinstance(gamma_range, float):
            self.a, self.b = 1-gamma_range, 1+gamma_range

        self.p = p

    def __call__(self, img):

        random_gamma = (self.b - self.a) * np.random.random_sample() + self.a
        if np.random.random_sample() <= self.p:
            return TF.adjust_gamma(img, random_gamma)
        else:
            return img

class RandomRotation():
    """ Random rotation
    """

    def __init__(self, angle_range, p=0.5):

        if isinstance(angle_range, tuple):
            self.a, self.b = angle_range
        if isinstance(angle_range, float) or isinstance(angle_range, int):
            self.a, self.b = -1*angle_range, angle_range

        self.p = p

    def __call__(self, img):

        random_angle = (self.b - self.a) * np.random.random_sample() + self.a

        if np.random.random_sample() <= self.p:
            return TF.rotate(
                img,
                angle = random_angle,
                interpolation = TF.InterpolationMode.BILINEAR,
                fill = 0,
            )
        else:
            return img