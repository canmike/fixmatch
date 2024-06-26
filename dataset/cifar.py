import logging
import math

import numpy as np
from PIL import Image
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import Dataset
import torch
import os
from PIL import Image

from .randaugment import RandAugmentMC

logger = logging.getLogger(__name__)

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)
cifar100_mean = (0.5071, 0.4867, 0.4408)
cifar100_std = (0.2675, 0.2565, 0.2761)
pss_mean = (0.52661989, 0.42101473, 0.34587943)
pss_std = (0.28316404, 0.2779676, 0.28667751)
normal_mean = (0.5, 0.5, 0.5)
normal_std = (0.5, 0.5, 0.5)


def get_pss(args, root):
    transform_labeled = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(
                size=32, padding=int(32 * 0.125), padding_mode="reflect"
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=pss_mean, std=pss_std),
        ]
    )
    transform_val = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=pss_mean, std=pss_std),
        ]
    )

    train_labeled_dataset = datasets.ImageFolder(
        root="C:\\Users\\can.michael\\Desktop\\others\\SSL\\FixMatch\\FixMatch-pytorch\\data\\pizza_steak_sushi\\20label\\train",
        transform=transform_labeled,
        target_transform=None,
    )

    train_unlabeled_dataset = PSS_Unlabeled(
        img_dir="C:\\Users\\can.michael\\Desktop\\others\SSL\\FixMatch\\FixMatch-pytorch\\data\\pizza_steak_sushi\\20label\\unlabeled",
        transform=TransformFixMatch(mean=pss_mean, std=pss_std),
    )

    test_dataset = datasets.ImageFolder(
        root="C:\\Users\\can.michael\\Desktop\\others\\SSL\\FixMatch\\FixMatch-pytorch\\data\\pizza_steak_sushi\\20label\\test",
        transform=transform_val,
        target_transform=None,
    )

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset


def get_cifar10(args, root):
    transform_labeled = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(
                size=32, padding=int(32 * 0.125), padding_mode="reflect"
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=cifar10_mean, std=cifar10_std),
        ]
    )
    transform_val = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=cifar10_mean, std=cifar10_std),
        ]
    )
    base_dataset = datasets.CIFAR10(root, train=True, download=True)

    train_labeled_idxs, train_unlabeled_idxs = x_u_split(args, base_dataset.targets)

    train_labeled_dataset = CIFAR10SSL(
        root, train_labeled_idxs, train=True, transform=transform_labeled
    )

    train_unlabeled_dataset = CIFAR10SSL(
        root,
        train_unlabeled_idxs,
        train=True,
        transform=TransformFixMatch(mean=cifar10_mean, std=cifar10_std),
    )

    test_dataset = datasets.CIFAR10(
        root, train=False, transform=transform_val, download=False
    )

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset


def get_cifar100(args, root):

    transform_labeled = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(
                size=32, padding=int(32 * 0.125), padding_mode="reflect"
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=cifar100_mean, std=cifar100_std),
        ]
    )

    transform_val = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=cifar100_mean, std=cifar100_std),
        ]
    )

    base_dataset = datasets.CIFAR100(root, train=True, download=True)

    train_labeled_idxs, train_unlabeled_idxs = x_u_split(args, base_dataset.targets)

    train_labeled_dataset = CIFAR100SSL(
        root, train_labeled_idxs, train=True, transform=transform_labeled
    )

    train_unlabeled_dataset = CIFAR100SSL(
        root,
        train_unlabeled_idxs,
        train=True,
        transform=TransformFixMatch(mean=cifar100_mean, std=cifar100_std),
    )

    test_dataset = datasets.CIFAR100(
        root, train=False, transform=transform_val, download=False
    )

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset


def x_u_split(args, labels):
    label_per_class = args.num_labeled // args.num_classes
    labels = np.array(labels)
    labeled_idx = []
    # unlabeled data: all data (https://github.com/kekmodel/FixMatch-pytorch/issues/10)
    unlabeled_idx = np.array(range(len(labels)))
    for i in range(args.num_classes):
        idx = np.where(labels == i)[0]
        idx = np.random.choice(idx, label_per_class, False)
        labeled_idx.extend(idx)
    labeled_idx = np.array(labeled_idx)
    assert len(labeled_idx) == args.num_labeled

    if args.expand_labels or args.num_labeled < args.batch_size:
        num_expand_x = math.ceil(args.batch_size * args.eval_step / args.num_labeled)
        labeled_idx = np.hstack([labeled_idx for _ in range(num_expand_x)])
    np.random.shuffle(labeled_idx)
    return labeled_idx, unlabeled_idx


class TransformFixMatch(object):
    def __init__(self, mean, std):
        self.weak = transforms.Compose(
            [
                transforms.Resize((32, 32)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(
                    size=32,
                    padding=int(32 * 0.125),
                    padding_mode="reflect",
                ),
            ]
        )
        self.strong = transforms.Compose(
            [
                transforms.Resize((32, 32)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(
                    size=32, padding=int(32 * 0.125), padding_mode="reflect"
                ),
                RandAugmentMC(n=2, m=10),
            ]
        )
        self.normalize = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
        )

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong)


class PSS_Unlabeled(Dataset):
    def __init__(self, img_dir, transform):
        self.ids = os.listdir(img_dir)
        self.ids.sort()

        self.images_fps = [os.path.join(img_dir, image_id) for image_id in self.ids]

        self.transform = transform

    def __len__(self):
        return len(self.images_fps)

    def __getitem__(self, i):
        image = Image.open(self.images_fps[i])
        img_transformed = self.transform(image)

        return img_transformed
        # img_transform = transforms.Compose([transforms.PILToTensor()])

        # return img_transform(img_transformed)


class CIFAR10SSL(datasets.CIFAR10):
    def __init__(
        self,
        root,
        indexs,
        train=True,
        transform=None,
        target_transform=None,
        download=False,
    ):
        super().__init__(
            root,
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class CIFAR100SSL(datasets.CIFAR100):
    def __init__(
        self,
        root,
        indexs,
        train=True,
        transform=None,
        target_transform=None,
        download=False,
    ):
        super().__init__(
            root,
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


DATASET_GETTERS = {"cifar10": get_cifar10, "cifar100": get_cifar100, "pss": get_pss}
