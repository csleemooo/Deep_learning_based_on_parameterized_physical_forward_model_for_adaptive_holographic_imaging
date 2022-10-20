import torch
from torch.utils.data import Dataset
from typing import Any, Callable, List, Optional, Tuple
import os
import numpy as np
from PIL import Image
import platform

np.random.seed(777)
os_name = platform.system().lower()
class Holo_Recon_Dataloader(Dataset):

    def __init__(self,
                 root: str,
                 data_type,
                 image_set: str="train",
                 transform: Optional[Callable] = None,
                 ratio: float = None,
                 train_type: str = None,
                 holo_list: list=None,
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

        if ratio is not None:
            dat_num = len(self.data_list[0])

            self.ratio_idx_set = np.arange(dat_num)
            np.random.shuffle(self.ratio_idx_set)

            self.ratio_idx_set = self.ratio_idx_set[:int(dat_num*ratio)]
            self.data_list = self.data_list[:, self.ratio_idx_set]

        self.data_num = len(self.data_list[0])

    def __len__(self) -> int:
        return self.data_num

    def __getitem__(self, index: int):

        if 'holography' in self.data_type:
            pth = os.path.join(self.root_list[0], self.data_list[0][index])
            
            if os_name == 'windows':
                distance =self.data_list[0][index].split("\\")[0]
            else:
                distance =self.data_list[0][index].split("/")[0]
                
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
