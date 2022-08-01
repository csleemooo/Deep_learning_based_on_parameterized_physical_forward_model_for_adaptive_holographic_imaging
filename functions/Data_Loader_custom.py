import torch
from torch.utils.data import Dataset
from typing import Any, Callable, List, Optional, Tuple
import os
import numpy as np
from PIL import Image

np.random.seed(777)

class Holo_Recon_Dataloader(Dataset):

    def __init__(self,
                 root: str,
                 data_type,
                 image_set: str="train",
                 transform: Optional[Callable] = None,
                 seed: int = None,
                 ratio: float = None,
                 train_type: str = None,
                 holo_list: list=None,
                 sort: bool=None,
                 return_distance: bool=None,
    ) -> None:

        self.transform = transform
        self.data_type = data_type
        self.return_distance = return_distance
        self.data_list=[]
        self.root_list=[]

        for i in self.data_type:
            tmp_root = os.path.join(root, image_set, i)

            self.root_list.append(tmp_root)

            if i =='holography':
                self.data_list.append([])
                dist_list = os.listdir(tmp_root)
                dist_list = sorted([float(i) if '.' in i else int(i) for i in dist_list])

                for dl in dist_list:
                    if dl in holo_list:
                        # self.data_list[0].extend([os.path.join('{:0<4}'.format(dl), j) for j in os.listdir(os.path.join(tmp_root, '{:0<4}'.format(dl)))])
                        self.data_list[0].extend([os.path.join(str(dl), j) for j in os.listdir(os.path.join(tmp_root, str(dl)))])

            else:
                self.data_list.append(os.listdir(tmp_root))

        self.data_list = np.array(self.data_list)

        if seed is not None:
            if train_type == "train":
                self.seed_idx_set =np.array([i for i in range(len(self.data_list[0])) if not i%10==seed])
            elif train_type == "val":
                self.seed_idx_set = np.arange(seed, len(self.data_list[0]), 10)

            self.data_list = self.data_list[:, self.seed_idx_set]

        if ratio is not None:
            dat_num = len(self.data_list[0])

            self.ratio_idx_set = np.arange(dat_num)
            np.random.shuffle(self.ratio_idx_set)

            self.ratio_idx_set = self.ratio_idx_set[:int(dat_num*ratio)]
            self.data_list = self.data_list[:, self.ratio_idx_set]

        if sort is not None:
            tmp = [int(i.split('\\')[-1].split('.mat')[0].replace('holography', '')) for i in np.ravel(self.data_list)]
            sort_idx = np.argsort(tmp)
            self.data_list = self.data_list[:, sort_idx]

        self.data_num = len(self.data_list[0])

    def __len__(self) -> int:
        return self.data_num

    def __getitem__(self, index: int):

        if 'holography' in self.data_type:
            pth = os.path.join(self.root_list[0], self.data_list[0][index])
            distance =self.data_list[0][index].split("\\")[0]
            distance = float(distance) if '.' in distance else int(distance)
            holo = self.load_matfile(pth)['holography']

            if self.return_distance:
                if self.transform is not None:
                    holo = self.transform(holo)
                    distance = torch.Tensor([distance])

                return holo, distance
            else:
                if self.transform is not None:
                    holo = self.transform(holo)

                return holo

        else:
            pth = os.path.join(self.root_list[0], self.data_list[0][index])
            gt_amplitude = self.load_matfile(pth)['gt_amplitude']

            pth = os.path.join(self.root_list[1], self.data_list[1][index])
            gt_phase = self.load_matfile(pth)['gt_phase']

            if self.transform is not None:
                gt_amplitude = self.transform(gt_amplitude)
                gt_phase = self.transform(gt_phase)

            return gt_amplitude, gt_phase

    def load_matfile(self, path):
        import scipy.io as sio
        data = sio.loadmat(path)
        return data


class Holo_Recon_Dataloader_style(Dataset):

    def __init__(self,
                 root: str,
                 data_type,
                 image_set: str="train",
                 transform: Optional[Callable] = None,
                 seed: int = None,
                 ratio: float = None,
                 train_type: str = None,
                 holo_list: list=None,
                 sort: bool=None,
                 return_distance: bool=None,
    ) -> None:

        self.transform = transform
        self.data_type = data_type
        self.return_distance = return_distance
        self.data_list=[]
        self.root_list=[]
        self.holo_list = holo_list

        for i in self.data_type:
            tmp_root = os.path.join(root, image_set, i)

            self.root_list.append(tmp_root)

            if i =='holography':
                self.data_list.append([])
                dist_list = os.listdir(tmp_root)
                dist_list = sorted([float(i) if '.' in i else int(i) for i in dist_list])

                for dl in dist_list:
                    if dl in holo_list:
                        self.data_list[0].extend([os.path.join('{:0<4}'.format(dl), j) for j in os.listdir(os.path.join(tmp_root, '{:0<4}'.format(dl)))])

            else:
                self.data_list.append(os.listdir(tmp_root))

        self.data_list = np.array(self.data_list)

        if seed is not None:
            if train_type == "train":
                self.seed_idx_set =np.array([i for i in range(len(self.data_list[0])) if not i%10==seed])
            elif train_type == "val":
                self.seed_idx_set = np.arange(seed, len(self.data_list[0]), 10)

            self.data_list = self.data_list[:, self.seed_idx_set]

        if ratio is not None:
            dat_num = len(self.data_list[0])

            self.ratio_idx_set = np.arange(dat_num)
            np.random.shuffle(self.ratio_idx_set)

            self.ratio_idx_set = self.ratio_idx_set[:int(dat_num*ratio)]
            self.data_list = self.data_list[:, self.ratio_idx_set]

        if sort is not None:
            tmp = [int(i.split('\\')[-1].split('.mat')[0].replace('holography', '')) for i in np.ravel(self.data_list)]
            sort_idx = np.argsort(tmp)
            self.data_list = self.data_list[:, sort_idx]

        self.data_num = len(self.data_list[0])

    def __len__(self) -> int:
        return self.data_num

    def __getitem__(self, index: int):

        if 'holography' in self.data_type:
            pth = os.path.join(self.root_list[0], self.data_list[0][index])
            distance =self.data_list[0][index].split("\\")[0]
            distance = float(distance) if '.' in distance else int(distance)

            distnace_style = [i for i in self.holo_list if i != distance][np.random.randint(0, len(self.holo_list)-1)]
            pth_style = os.listdir(os.path.join(self.root_list[0], '{:0<4}'.format(distnace_style)))
            pth_style = os.path.join(self.root_list[0], '{:0<4}'.format(distnace_style), pth_style[np.random.randint(0, len(pth_style))])

            holo = self.load_matfile(pth)['holography']
            holo_style = self.load_matfile(pth_style)['holography']

            if self.return_distance:
                if self.transform is not None:
                    holo = self.transform(holo)
                    holo_style = self.transform(holo_style)

                    distance = torch.Tensor([distance])
                    distance_style = torch.Tensor([distnace_style])

                return [holo, distance], [holo_style, distance_style]
            else:
                if self.transform is not None:
                    holo = self.transform(holo)

                return holo

        else:
            pth = os.path.join(self.root_list[0], self.data_list[0][index])
            gt_amplitude = self.load_matfile(pth)['gt_amplitude']

            pth = os.path.join(self.root_list[1], self.data_list[1][index])
            gt_phase = self.load_matfile(pth)['gt_phase']

            if self.transform is not None:
                gt_amplitude = self.transform(gt_amplitude)
                gt_phase = self.transform(gt_phase)

            return gt_amplitude, gt_phase

    def load_matfile(self, path):
        import scipy.io as sio
        data = sio.loadmat(path)
        return data

class Holo_distance_loader(Dataset):

    def __init__(self,
                 root: str,
                 data_type,
                 image_set: str="train",
                 transform: Optional[Callable] = None,
                 holo_list: list=None,
                 sort: bool=None,
                 return_distance: bool=None,
    ) -> None:

        self.transform = transform
        self.data_type = data_type
        self.return_distance = return_distance
        self.data_list=[]
        self.root_list=[]
        self.dist_list=[]

        tmp_root = os.path.join(root, image_set, self.data_type)

        self.root_list.append(tmp_root)

        if self.data_type =='holography':
            self.data_list.append([])
            dist_list = os.listdir(tmp_root)
            dist_list = sorted([float(i) if '.' in i else int(i) for i in dist_list])

            for dl in dist_list:
                if dl in holo_list:
                    self.data_list[0].extend([os.path.join(str(dl), j) for j in os.listdir(os.path.join(tmp_root, str(dl)))])
                    self.dist_list.extend([dl]*len(os.listdir(os.path.join(tmp_root, str(dl)))))


        self.data_list = np.array(self.data_list)
        self.dist_list = np.array(self.dist_list)
        self.dist_class = {i:c_idx for c_idx, i in enumerate(np.unique(self.dist_list))}
        self.class_num = len(self.dist_class)

        if sort is not None:
            tmp = [int(i.split('\\')[-1].split('.mat')[0].replace('holography', '')) for i in np.ravel(self.data_list)]
            sort_idx = np.argsort(tmp)
            self.data_list = self.data_list[:, sort_idx]
            self.dist_list = self.dist_list[:, sort_idx]

        self.data_num = len(self.data_list[0])

    def __len__(self) -> int:
        return self.data_num

    def __getitem__(self, index: int):

        pth = os.path.join(self.root_list[0], self.data_list[0][index])
        holo = self.load_matfile(pth)['holography']
        distance =self.dist_list[index]
        distance_class = [self.dist_class[distance]]

        if self.transform is not None:
            holo = self.transform(holo)
            distance_class = torch.Tensor(distance_class)

        if self.return_distance:
            return holo, distance_class, distance
        else:
            return holo, distance_class

    def load_matfile(self, path):
        import scipy.io as sio
        data = sio.loadmat(path)
        return data

class Holo_Recon_Dataloader_supervised(Dataset):

    def __init__(self,
                 root: str,
                 data_type,
                 image_set: str="train",
                 transform: Optional[Callable] = None,
                 seed: int = None,
                 ratio: float = None,
                 train_type: str = None,
                 holo_list: list=None,
                 sort: bool=None,
                 return_distance: bool=None,
    ) -> None:

        self.transform = transform
        self.data_type = data_type
        self.return_distance = return_distance
        self.data_list=[]
        self.root_list=[]

        tmp_root = os.path.join(root, image_set, 'holography')

        self.root_list.append(tmp_root)

        self.data_list.append([])
        dist_list = os.listdir(tmp_root)
        dist_list = sorted([int(i) for i in dist_list])

        for dl in dist_list:
            if dl in holo_list:
                self.data_list[0].extend([os.path.join(str(dl), j) for j in os.listdir(os.path.join(tmp_root, str(dl)))])

        self.data_list = np.array(self.data_list)

        if seed is not None:
            if train_type == "train":
                self.seed_idx_set =np.array([i for i in range(len(self.data_list[0])) if not i%10==seed])
            elif train_type == "val":
                self.seed_idx_set = np.arange(seed, len(self.data_list[0]), 10)

            self.data_list = self.data_list[:, self.seed_idx_set]

        if ratio is not None:
            dat_num = len(self.data_list[0])

            self.ratio_idx_set =np.arange(dat_num)
            np.random.shuffle(self.ratio_idx_set)

            self.ratio_idx_set = self.ratio_idx_set[:int(dat_num*ratio)]
            self.data_list = self.data_list[:, self.ratio_idx_set]

        if sort is not None:
            tmp = [int(i.split('\\')[-1].split('.mat')[0].replace('holography', '')) for i in np.ravel(self.data_list)]
            sort_idx = np.argsort(tmp)
            self.data_list = self.data_list[:, sort_idx]

        self.data_num = len(self.data_list[0])
        self.root_list = '\\'.join(self.root_list[0].split("\\")[:-1])

    def __len__(self) -> int:
        return self.data_num

    def __getitem__(self, index: int):


        data_num = self.data_list[0][index].split('\\')[-1].split('.')[0].replace('holography', '')

        pth = os.path.join(self.root_list, 'holography', self.data_list[0][index])
        holo = self.load_matfile(pth)['holography']

        pth = os.path.join(self.root_list, 'gt_amplitude', 'gt_amplitude'+data_num+'.mat')
        gt_amplitude = self.load_matfile(pth)['gt_amplitude']

        pth = os.path.join(self.root_list, 'gt_phase', 'gt_phase'+data_num+'.mat')
        gt_phase = self.load_matfile(pth)['gt_phase']

        if self.transform is not None:
            gt_amplitude = self.transform(gt_amplitude)
            gt_phase = self.transform(gt_phase)
            holo = self.transform(holo)

        return holo, gt_amplitude, gt_phase



    def load_matfile(self, path):
        import scipy.io as sio
        data = sio.loadmat(path)
        return data


class Simulation_Dataloader(Dataset):

    def __init__(self,
                 root: str,
                 data_type,
                 image_set: str="train",
                 transform: Optional[Callable] = None,
                 seed: int = None,
                 ratio: float = None,
                 train_type: str = None,
                 holo_list: list=None,
                 sort: bool=None
    ) -> None:

        self.transform = transform
        self.data_type = data_type

        self.root = os.path.join(root, image_set, data_type)

        self.data_list = []
        for p_name in os.listdir(self.root):
            self.data_list.extend([os.path.join(p_name, p) for p in os.listdir(os.path.join(self.root, p_name))])


        if ratio is not None:
            dat_num = len(self.data_list)

            self.ratio_idx_set =np.arange(dat_num)
            np.random.shuffle(self.ratio_idx_set)

            self.ratio_idx_set = self.ratio_idx_set[:int(dat_num*ratio)]
            self.data_list = self.data_list[self.ratio_idx_set]

        self.data_num = len(self.data_list)

    def __len__(self) -> int:
        return self.data_num

    def __getitem__(self, index: int):

        data = Image.open(os.path.join(self.root, self.data_list[index]))

        if self.transform is not None:
            data = self.transform(data)

        return data
