import h5py
import SimpleITK as sitk
import torch
import itertools
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
import glob
import os
import monai.transforms as aug
import nibabel as nib

class LAHeart(Dataset):
    """ LA Dataset """
    def __init__(self, base_dir=None, split='train', num=None, transform=None):
        self._base_dir = base_dir
        self.transform = transform
        self.sample_list = []
       
        print(self._base_dir)

        if split=='train':
            with open(self._base_dir+'/../train.list', 'r') as f:
                self.image_list = f.readlines()
        elif split == 'test':
            with open(self._base_dir+'/../test.list', 'r') as f:
                self.image_list = f.readlines()
        self.image_list = [item.replace('\n', '') for item in self.image_list]
        if num is not None:
            self.image_list = self.image_list[:num]
        print("total {} samples".format(len(self.image_list)))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        h5f = h5py.File(self._base_dir + "/" + image_name + "/mri_norm2.h5", 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        label_full = h5f['label'][:]#h5f['label_full'][:]
        sample = {'image': image, 'label': label, 'label_full': label_full}
        if self.transform:
            sample = self.transform(sample)
        return sample

class FemurDSet(Dataset):
    def __init__(self, base_dir=None, split='train', num=None, transform=None, label_suffix='-label-thresholded', verbose=False):
        self._base_dir = os.path.join(base_dir, split)
        self.transform = transform
        self.label_suffix = label_suffix

        # exclude the label files cuz thats images
        self.image_list = sorted(
            [f for f in glob.glob(os.path.join(self._base_dir, "*.nii.gz")) if label_suffix not in f]
        )

        if num is not None:
            self.image_list = self.image_list[:num]

        if verbose:
            print(f"Total {len(self.image_list)} samples found in {self._base_dir}")

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_path = self.image_list[idx]
        label_path = image_path.replace(".nii.gz", f"{self.label_suffix}.nii.gz")

        # Load NIfTI data
        image = nib.load(image_path).get_fdata().astype(np.float32)
        label = nib.load(label_path).get_fdata().astype(np.uint8)
        label = (label == 1).astype(np.uint8)

        sample = {'image': image, 'label': label, 'label_full': label}

        if self.transform:
            sample = self.transform(sample)

        return sample
    
# class FemurDSet(Dataset):
#     """ LA/Femur Dataset """
#     def __init__(self, base_dir=None, split='train', num=None, transform=None):
#         self._base_dir = base_dir
#         self.transform = transform
#         self.sample_list = []

#         self.image_list = glob.glob(os.path.join(base_dir, split, "*.h5"))
#         self.image_list.sort()
#         # print(len(self.image_list))
#         if num is not None:
#             self.image_list = self.image_list[:num]
#         print("total {} samples".format(len(self.image_list)))

#     def __len__(self):
#         return len(self.image_list)

#     def __getitem__(self, idx):
#         # image_name = self.image_list[idx]
#         h5f = h5py.File(self.image_list[idx], 'r')
#         image = h5f['image'][:]
#         label = h5f['label'][:]
#         label_full = h5f['label'][:]#h5f['label_full'][:]
#         # print(image.shape, label.shape)
#         sample = {'image': image, 'label': label, 'label_full': label_full}
#         if self.transform:
#             sample = self.transform(sample)
#         return sample

class CenterCrop(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label, label_full = sample['image'], sample['label'], sample['label_full']

        # pad the sample if necessary
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            label_full = np.pad(label_full, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

        (w, h, d) = image.shape

        w1 = int(round((w - self.output_size[0]) / 2.))
        h1 = int(round((h - self.output_size[1]) / 2.))
        d1 = int(round((d - self.output_size[2]) / 2.))

        image = image[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        label = label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        label_full = label_full[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        return {'image': image, 'label': label, 'label_full': label_full}


class RandomCrop(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label, label_full = sample['image'], sample['label'], sample['label_full']

        # pad the sample if necessary
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            label_full = np.pad(label_full, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

        (w, h, d) = image.shape
        w1 = np.random.randint(0, w - self.output_size[0])
        h1 = np.random.randint(0, h - self.output_size[1])
        d1 = np.random.randint(0, d - self.output_size[2])

        image = image[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        label = label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        label_full = label_full[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        return {'image': image, 'label': label, 'label_full': label_full}


class RandomRotFlip(object):
    def __call__(self, sample):
        image, label, label_full = sample['image'], sample['label'], sample['label_full']
        k = np.random.randint(0, 4)
        image = np.rot90(image, k)
        label = np.rot90(label, k)
        label_full = np.rot90(label_full, k)
        axis = np.random.randint(0, 2)
        image = np.flip(image, axis=axis).copy()
        label = np.flip(label, axis=axis).copy()
        label_full = np.flip(label_full, axis=axis).copy()

        return {'image': image, 'label': label, 'label_full': label_full}


class ToTensor(object):
    def __call__(self, sample):
        image = sample['image']
        image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2]).astype(np.float32)
        if 'onehot_label' in sample:
            return {'image': torch.from_numpy(image), 'label': torch.from_numpy(sample['label']).long(), 'label_full': torch.from_numpy(sample['label_full']).long(),
                    'onehot_label': torch.from_numpy(sample['onehot_label']).long()}
        else:
            return {'image': torch.from_numpy(image), 'label': torch.from_numpy(sample['label']).long(), 'label_full': torch.from_numpy(sample['label_full']).long()}


class TwoStreamBatchSampler(Sampler):
    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in zip(grouper(primary_iter, self.primary_batch_size),
                    grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    args = [iter(iterable)] * n
    return zip(*args)


class LAHeart_FixMatch(Dataset):
    """ LA Dataset """
    def __init__(self, base_dir=None, split='train', num=None, transform=None):
        self._base_dir = base_dir
        self.transform = transform
        self.sample_list = []

        if split=='train':
            with open(self._base_dir+'/../train.list', 'r') as f:
                self.image_list = f.readlines()
        elif split == 'test':
            with open(self._base_dir+'/../test.list', 'r') as f:
                self.image_list = f.readlines()
        self.image_list = [item.replace('\n', '') for item in self.image_list]
        if num is not None:
            self.image_list = self.image_list[:num]
        print("total {} samples".format(len(self.image_list)))
        
        self.augment_list = [
            aug.RandShiftIntensity(offsets=10, prob=0.5),
            aug.RandAdjustContrast(gamma=(0.5, 8), prob=0.5),
            aug.RandHistogramShift(prob=0.5),
            aug.RandGaussianSharpen(prob=0.5),
            aug.RandGaussianNoise(prob=0.5),
            aug.RandAffine(prob=0.5)
        ]
        
        self.aug_transform = aug.SomeOf(self.augment_list, num_transforms=2)
        self.cutout = aug.RandCoarseDropout(prob=1,holes=10, max_holes=30,spatial_size=5, max_spatial_size=10)
        
    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        h5f = h5py.File(self._base_dir + "/" + image_name + "/mri_norm2.h5", 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        label_full = h5f['label'][:]#h5f['label_full'][:]
        sample = {'image': image, 'label': label, 'label_full': label_full}
        if self.transform:
            sample = self.transform(sample)
            image_strong = self.cutout(self.aug_transform(sample['image']))
            sample['image_strong'] = image_strong
        return sample


class FixMatchDSet(Dataset):
    """ LA Dataset """
    def __init__(self, base_dir=None, split='train', num=None, transform=None):
        self._base_dir = base_dir
        self.transform = transform
        self.sample_list = []

        self.image_list = glob.glob(os.path.join(base_dir, split, "*.h5"))
        self.image_list.sort()
        # print(len(self.image_list))
        if num is not None:
            self.image_list = self.image_list[:num]
        print("total {} samples".format(len(self.image_list)))
        
        self.augment_list = [
            aug.RandShiftIntensity(offsets=10, prob=0.5),
            aug.RandAdjustContrast(gamma=(0.5, 8), prob=0.5),
            aug.RandHistogramShift(prob=0.5),
            aug.RandGaussianSharpen(prob=0.5),
            aug.RandGaussianNoise(prob=0.5),
            aug.RandAffine(prob=0.5)
        ]
        
        self.aug_transform = aug.SomeOf(self.augment_list, num_transforms=2)
        self.cutout = aug.RandCoarseDropout(prob=1,holes=10, max_holes=30,spatial_size=5, max_spatial_size=10)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        # image_name = self.image_list[idx]
        h5f = h5py.File(self.image_list[idx], 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        label_full = h5f['label'][:]#h5f['label_full'][:]
        # print(image.shape, label.shape)
        sample = {'image': image, 'label': label, 'label_full': label_full}
        if self.transform:
            sample = self.transform(sample)
            image_strong = self.cutout(self.aug_transform(sample['image']))
            sample['image_strong'] = image_strong
        return sample





































# import h5py
# import torch
# import itertools
# import numpy as np
# from torch.utils.data import Dataset
# from torch.utils.data.sampler import Sampler
# import glob
# import os
# import monai.transforms as aug

# class LAHeart(Dataset):
#     """ LA Dataset """
#     def __init__(self, base_dir=None, split='train', num=None, transform=None):
#         self._base_dir = base_dir
#         self.transform = transform
#         self.sample_list = []
       
#         print(self._base_dir)

#         if split=='train':
#             with open(self._base_dir+'/../train.list', 'r') as f:
#                 self.image_list = f.readlines()
#         elif split == 'test':
#             with open(self._base_dir+'/../test.list', 'r') as f:
#                 self.image_list = f.readlines()
#         self.image_list = [item.replace('\n', '') for item in self.image_list]
#         if num is not None:
#             self.image_list = self.image_list[:num]
#         print("total {} samples".format(len(self.image_list)))

#     def __len__(self):
#         return len(self.image_list)

#     def __getitem__(self, idx):
#         image_name = self.image_list[idx]
#         h5f = h5py.File(self._base_dir + "/" + image_name + "/mri_norm2.h5", 'r')
#         image = h5f['image'][:]
#         label = h5f['label'][:]
#         label_full = h5f['label'][:]#h5f['label_full'][:]
#         sample = {'image': image, 'label': label, 'label_full': label_full}
#         if self.transform:
#             sample = self.transform(sample)
#         return sample


# class FemurDSet(Dataset):
#     """ LA/Femur Dataset """
#     def __init__(self, base_dir=None, split='train', num=None, transform=None):
#         self._base_dir = base_dir
#         self.transform = transform
#         self.sample_list = []

#         self.image_list = glob.glob(os.path.join(base_dir, split, "*.h5"))
#         self.image_list.sort()
#         # print(len(self.image_list))
#         if num is not None:
#             self.image_list = self.image_list[:num]
#         print("total {} samples".format(len(self.image_list)))

#     def __len__(self):
#         return len(self.image_list)

#     def __getitem__(self, idx):
#         # image_name = self.image_list[idx]
#         h5f = h5py.File(self.image_list[idx], 'r')
#         image = h5f['image'][:]
#         label = h5f['label'][:]
#         label_full = h5f['label'][:]#h5f['label_full'][:]
#         # print(image.shape, label.shape)
#         sample = {'image': image, 'label': label, 'label_full': label_full}
#         if self.transform:
#             sample = self.transform(sample)
#         return sample

# class CenterCrop(object):
#     def __init__(self, output_size):
#         self.output_size = output_size

#     def __call__(self, sample):
#         image, label, label_full = sample['image'], sample['label'], sample['label_full']

#         # pad the sample if necessary
#         if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= \
#                 self.output_size[2]:
#             pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
#             ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
#             pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
#             image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
#             label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
#             label_full = np.pad(label_full, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

#         (w, h, d) = image.shape

#         w1 = int(round((w - self.output_size[0]) / 2.))
#         h1 = int(round((h - self.output_size[1]) / 2.))
#         d1 = int(round((d - self.output_size[2]) / 2.))

#         image = image[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
#         label = label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
#         label_full = label_full[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
#         return {'image': image, 'label': label, 'label_full': label_full}


# class RandomCrop(object):
#     def __init__(self, output_size):
#         self.output_size = output_size

#     def __call__(self, sample):
#         image, label, label_full = sample['image'], sample['label'], sample['label_full']

#         # pad the sample if necessary
#         if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= \
#                 self.output_size[2]:
#             pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
#             ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
#             pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
#             image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
#             label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
#             label_full = np.pad(label_full, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

#         (w, h, d) = image.shape
#         w1 = np.random.randint(0, w - self.output_size[0])
#         h1 = np.random.randint(0, h - self.output_size[1])
#         d1 = np.random.randint(0, d - self.output_size[2])

#         image = image[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
#         label = label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
#         label_full = label_full[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
#         return {'image': image, 'label': label, 'label_full': label_full}


# class RandomRotFlip(object):
#     def __call__(self, sample):
#         image, label, label_full = sample['image'], sample['label'], sample['label_full']
#         k = np.random.randint(0, 4)
#         image = np.rot90(image, k)
#         label = np.rot90(label, k)
#         label_full = np.rot90(label_full, k)
#         axis = np.random.randint(0, 2)
#         image = np.flip(image, axis=axis).copy()
#         label = np.flip(label, axis=axis).copy()
#         label_full = np.flip(label_full, axis=axis).copy()

#         return {'image': image, 'label': label, 'label_full': label_full}


# class ToTensor(object):
#     def __call__(self, sample):
#         image = sample['image']
#         image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2]).astype(np.float32)
#         if 'onehot_label' in sample:
#             return {'image': torch.from_numpy(image), 'label': torch.from_numpy(sample['label']).long(), 'label_full': torch.from_numpy(sample['label_full']).long(),
#                     'onehot_label': torch.from_numpy(sample['onehot_label']).long()}
#         else:
#             return {'image': torch.from_numpy(image), 'label': torch.from_numpy(sample['label']).long(), 'label_full': torch.from_numpy(sample['label_full']).long()}


# class TwoStreamBatchSampler(Sampler):
#     def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
#         self.primary_indices = primary_indices
#         self.secondary_indices = secondary_indices
#         self.secondary_batch_size = secondary_batch_size
#         self.primary_batch_size = batch_size - secondary_batch_size

#         assert len(self.primary_indices) >= self.primary_batch_size > 0
#         assert len(self.secondary_indices) >= self.secondary_batch_size > 0

#     def __iter__(self):
#         primary_iter = iterate_once(self.primary_indices)
#         secondary_iter = iterate_eternally(self.secondary_indices)
#         return (
#             primary_batch + secondary_batch
#             for (primary_batch, secondary_batch)
#             in zip(grouper(primary_iter, self.primary_batch_size),
#                     grouper(secondary_iter, self.secondary_batch_size))
#         )

#     def __len__(self):
#         return len(self.primary_indices) // self.primary_batch_size


# def iterate_once(iterable):
#     return np.random.permutation(iterable)


# def iterate_eternally(indices):
#     def infinite_shuffles():
#         while True:
#             yield np.random.permutation(indices)
#     return itertools.chain.from_iterable(infinite_shuffles())


# def grouper(iterable, n):
#     args = [iter(iterable)] * n
#     return zip(*args)


# class LAHeart_FixMatch(Dataset):
#     """ LA Dataset """
#     def __init__(self, base_dir=None, split='train', num=None, transform=None):
#         self._base_dir = base_dir
#         self.transform = transform
#         self.sample_list = []

#         if split=='train':
#             with open(self._base_dir+'/../train.list', 'r') as f:
#                 self.image_list = f.readlines()
#         elif split == 'test':
#             with open(self._base_dir+'/../test.list', 'r') as f:
#                 self.image_list = f.readlines()
#         self.image_list = [item.replace('\n', '') for item in self.image_list]
#         if num is not None:
#             self.image_list = self.image_list[:num]
#         print("total {} samples".format(len(self.image_list)))
        
#         self.augment_list = [
#             aug.RandShiftIntensity(offsets=10, prob=0.5),
#             aug.RandAdjustContrast(gamma=(0.5, 8), prob=0.5),
#             aug.RandHistogramShift(prob=0.5),
#             aug.RandGaussianSharpen(prob=0.5),
#             aug.RandGaussianNoise(prob=0.5),
#             aug.RandAffine(prob=0.5)
#         ]
        
#         self.aug_transform = aug.SomeOf(self.augment_list, num_transforms=2)
#         self.cutout = aug.RandCoarseDropout(prob=1,holes=10, max_holes=30,spatial_size=5, max_spatial_size=10)
        
#     def __len__(self):
#         return len(self.image_list)

#     def __getitem__(self, idx):
#         image_name = self.image_list[idx]
#         h5f = h5py.File(self._base_dir + "/" + image_name + "/mri_norm2.h5", 'r')
#         image = h5f['image'][:]
#         label = h5f['label'][:]
#         label_full = h5f['label'][:]#h5f['label_full'][:]
#         sample = {'image': image, 'label': label, 'label_full': label_full}
#         if self.transform:
#             sample = self.transform(sample)
#             image_strong = self.cutout(self.aug_transform(sample['image']))
#             sample['image_strong'] = image_strong
#         return sample


# class FixMatchDSet(Dataset):
#     """ LA Dataset """
#     def __init__(self, base_dir=None, split='train', num=None, transform=None):
#         self._base_dir = base_dir
#         self.transform = transform
#         self.sample_list = []

#         self.image_list = glob.glob(os.path.join(base_dir, split, "*.h5"))
#         self.image_list.sort()
#         # print(len(self.image_list))
#         if num is not None:
#             self.image_list = self.image_list[:num]
#         print("total {} samples".format(len(self.image_list)))
        
#         self.augment_list = [
#             aug.RandShiftIntensity(offsets=10, prob=0.5),
#             aug.RandAdjustContrast(gamma=(0.5, 8), prob=0.5),
#             aug.RandHistogramShift(prob=0.5),
#             aug.RandGaussianSharpen(prob=0.5),
#             aug.RandGaussianNoise(prob=0.5),
#             aug.RandAffine(prob=0.5)
#         ]
        
#         self.aug_transform = aug.SomeOf(self.augment_list, num_transforms=2)
#         self.cutout = aug.RandCoarseDropout(prob=1,holes=10, max_holes=30,spatial_size=5, max_spatial_size=10)

#     def __len__(self):
#         return len(self.image_list)

#     def __getitem__(self, idx):
#         # image_name = self.image_list[idx]
#         h5f = h5py.File(self.image_list[idx], 'r')
#         image = h5f['image'][:]
#         label = h5f['label'][:]
#         label_full = h5f['label'][:]#h5f['label_full'][:]
#         # print(image.shape, label.shape)
#         sample = {'image': image, 'label': label, 'label_full': label_full}
#         if self.transform:
#             sample = self.transform(sample)
#             image_strong = self.cutout(self.aug_transform(sample['image']))
#             sample['image_strong'] = image_strong
#         return sample