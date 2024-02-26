import torch.utils.data as data
from PIL import Image
import os
import os.path
import numpy as np
from typing import Any, Callable, Optional, Tuple
from .utils import download_url, check_integrity, verify_str_arg,noisify_instance_svhn,noisify
import scipy.io as sio
import torchvision.transforms as transforms

class SVHN(data.Dataset):
    """`SVHN <http://ufldl.stanford.edu/housenumbers/>`_ Dataset.
    Note: The SVHN dataset assigns the label `10` to the digit `0`. However, in this Dataset,
    we assign the label `0` to the digit `0` to be compatible with PyTorch loss functions which
    expect the class labels to be in the range `[0, C-1]`

    .. warning::

        This class needs `scipy <https://docs.scipy.org/doc/>`_ to load data from `.mat` format.

    Args:
        root (string): Root directory of dataset where directory
            ``SVHN`` exists.
        split (string): One of {'train', 'test', 'extra'}.
            Accordingly dataset is selected. 'extra' is Extra training set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """

    split_list = {
        'train': ["http://ufldl.stanford.edu/housenumbers/train_32x32.mat",
                  "train_32x32.mat", "e26dedcc434d2e4c54c9b2d4a06d8373"],
        'test': ["http://ufldl.stanford.edu/housenumbers/test_32x32.mat",
                 "test_32x32.mat", "eb5a983be6a315427106f1b164d9cef3"],
        'extra': ["http://ufldl.stanford.edu/housenumbers/extra_32x32.mat",
                  "extra_32x32.mat", "a93ce644f1a588dc4d68dda5feec44a7"]}
    filename = "train_32x32.mat"

    def __init__(
            self,
            root: str,
            split: str = "train",
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
            train=True,
            noise_type=None,
            noise_rate=0.2,
            random_state=0
    ):
        self.root = os.path.expanduser(root)
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        self.dataset='SVHN'
        self.noise_type=noise_type
        idx_each_class_noisy = [[] for i in range(10)]
        self.transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1201, 0.1231, 0.1052)),
        ])
        self.transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4524, 0.4525, 0.4690), (0.1225, 0.1283, 0.1144)),
        ])


        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        if train:
            split = "train"
            self.split = verify_str_arg(split, "split", tuple(self.split_list.keys()))
            self.url = self.split_list[split][0]
            self.filename = self.split_list[split][1]
            self.file_md5 = self.split_list[split][2]
            # reading(loading) mat file as array
            loaded_mat = sio.loadmat(os.path.join(self.root, self.filename))

            self.train_data = loaded_mat['X']
            # loading from the .mat file gives an np array of type np.uint8
            # converting to np.int64, so that we have a LongTensor after
            # the conversion from the numpy array
            # the squeeze is needed to obtain a 1D tensor
            self.train_labels = loaded_mat['y'].astype(np.int64).squeeze()

            # the svhn dataset assigns the class label "10" to the digit 0
            # this makes it inconsistent with several loss functions
            # which expect the class labels to be in the range [0, C-1]
            np.place(self.train_labels, self.train_labels == 10, 0)
            self.train_data = np.transpose(self.train_data, (3, 2, 0, 1))

            if noise_type != 'clean':
                if noise_type == "instance":
                    self.train_noisy_labels, self.actual_noise_rate = noisify_instance_svhn(self.train_data,
                                                                                       self.train_labels,
                                                                                       noise_rate=noise_rate)
                    print('over all noise rate is ', self.actual_noise_rate)

                    self.train_labels = np.asarray([[self.train_labels[i]] for i in range(len(self.train_labels))])

                    for i in range(len(self.train_labels)):
                        idx_each_class_noisy[self.train_noisy_labels[i]].append(i)
                    class_size_noisy = [len(idx_each_class_noisy[i]) for i in range(10)]
                    self.noise_prior = np.array(class_size_noisy) / sum(class_size_noisy)
                    print(f'The noisy data ratio in each class is {self.noise_prior}')

                    # self.train_noisy_labels = [i[0] for i in self.train_noisy_labels]

                    _train_labels = [i[0] for i in self.train_labels]

                    self.noise_or_not = np.transpose(self.train_noisy_labels) == np.transpose(_train_labels)
                else:
                    self.train_labels=np.asarray([[self.train_labels[i]] for i in range(len(self.train_labels))])
                    self.train_noisy_labels, self.actual_noise_rate = noisify(dataset=self.dataset, train_labels=self.train_labels, noise_type=noise_type, noise_rate=noise_rate, random_state=random_state)
                    self.train_noisy_labels=[i[0] for i in self.train_noisy_labels]
                    _train_labels=[i[0] for i in self.train_labels]
                    self.noise_or_not = np.transpose(self.train_noisy_labels)==np.transpose(_train_labels)

        else:
            split = "test"
            self.split = verify_str_arg(split, "split", tuple(self.split_list.keys()))
            self.url = self.split_list[split][0]
            self.filename = self.split_list[split][1]
            self.file_md5 = self.split_list[split][2]
            # reading(loading) mat file as array
            loaded_mat = sio.loadmat(os.path.join(self.root, self.filename))

            self.test_data = loaded_mat['X']
            # loading from the .mat file gives an np array of type np.uint8
            # converting to np.int64, so that we have a LongTensor after
            # the conversion from the numpy array
            # the squeeze is needed to obtain a 1D tensor
            self.test_labels = loaded_mat['y'].astype(np.int64).squeeze()

            # the svhn dataset assigns the class label "10" to the digit 0
            # this makes it inconsistent with several loss functions
            # which expect the class labels to be in the range [0, C-1]
            np.place(self.test_labels, self.test_labels == 10, 0)
            self.test_data = np.transpose(self.test_data, (3, 2, 0, 1))


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            if self.noise_type is not None:
                img, target = self.train_data[index], self.train_noisy_labels[index]
                img = Image.fromarray(np.transpose(img, (1, 2, 0)))
                img = self.transform_train(img)
            return img, target, index
        else:
            img, target = self.test_data[index], self.test_labels[index]
            img = Image.fromarray(np.transpose(img, (1, 2, 0)))
            img = self.transform_test(img)
            return img, target,index

    def __len__(self) -> int:
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def _check_integrity(self) -> bool:
        root = self.root
        md5 = self.split_list[self.split][2]
        fpath = os.path.join(root, self.filename)
        return check_integrity(fpath, md5)

    def download(self) -> None:
        md5 = self.split_list[self.split][2]
        download_url(self.url, self.root, self.filename, md5)

    def extra_repr(self) -> str:
        return "Split: {split}".format(**self.__dict__)
