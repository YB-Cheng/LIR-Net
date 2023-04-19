import gzip
import os

import numpy as np
import torch

from config import DATA_PATH


# ----- data transforms -----


class Permute(object):
    """构建（fbp，sino，target）三元组，输出要在实际应用处验证"""
    def __init__(self, perm):
        self.perm = perm

    def __call__(self, inputs):
        out = tuple([inputs[k] for k in self.perm])
        return out


# ----- datasets -----
class CTDataset(torch.utils.data.Dataset):
    """ AAPM Computed Tomography Challenge dataset.
        AAPM挑战赛数据集

    Loads (fbp, sinogram, target) data from a single data batch file.
    从一个单一的数据目录下加载（fbp, sinogram, target）

    参数说明：
    ----------
    subset : string
        'train', 'val', 或者 'test' 之一，或者有效的子目录路径
        确定要搜索数据文件的子目录。
    batch : int
        要加载的数据批的编号 训练集[1,2,3,4]之一 （训练集有四批数据）
        测试集和训练集都只有一批数据，即只有1
    folds : int
        用于数据拆分的折叠数（例如用于交叉验证）默认10
    num_fold: int or list
        要使用的当前折叠数。[0，…，folds-1]之一，可以使用列表创建多个折数，默认0
    leave_out : bool
        保留指定折数，训练数据设置true，验证数据为false
    transform : callable
        用于预处理的其他数据转换（默认无）
        设备：torch.device
        放置数据的设备（例如CPU、GPU cuda说明符）。
        （默认无）

    """

    def __init__(
        self,
        subset,
        batch,
        folds=10,
        num_fold=0,
        leave_out=True,
        transform=None,
        device=None,
    ):
        # choose directory according to subset
        if subset == "train":
            path = os.path.join(DATA_PATH, "training_data")
        elif subset == "val":
            path = os.path.join(DATA_PATH, "validation_data")
        elif subset == "test":
            path = os.path.join(DATA_PATH, "test_data")
        else:
            path = os.path.join(DATA_PATH, subset)

        self.transform = transform
        self.device = device

        # load data files
        self.sinogram = np.load(os.path.join(path, "TargetSin_batch{}.npy".format(batch)))
        self.fbp = np.load(os.path.join(path, "TargetImg_batch{}.npy".format(batch)))

        if not subset == "val" and not subset == "test":
            self.phantom = np.load(os.path.join(path, "SparseSin_batch{}.npy".format(batch)))
        else:
            self.phantom = 0.0 * self.fbp  #测试集和验证集不存在label

        assert self.phantom.shape[0] == self.sinogram.shape[0]#检查数据在数量上相对应
        assert self.phantom.shape[0] == self.fbp.shape[0]

        # split dataset for cross validation拆分数据以交叉验证
        fold_len = self.phantom.shape[0] // folds#做除法并向下取整
        if not isinstance(num_fold, list):
            num_fold = [num_fold]
        p_list, s_list, f_list = [], [], []
        for cur_fold in range(folds):
            il = cur_fold * fold_len
            ir = il + fold_len
            if leave_out ^ (cur_fold in num_fold):
                p_list.append(self.phantom[il:ir])
                s_list.append(self.sinogram[il:ir])
                f_list.append(self.fbp[il:ir])
        self.phantom = np.concatenate(p_list, axis=0)
        self.sinogram = np.concatenate(s_list, axis=0)
        self.fbp = np.concatenate(f_list, axis=0)

        # transform numpy to torch tensor
        self.phantom = torch.tensor(self.phantom, dtype=torch.float)
        self.sinogram = torch.tensor(self.sinogram, dtype=torch.float)
        self.fbp = torch.tensor(self.fbp, dtype=torch.float)

    def __len__(self):
        return self.phantom.shape[0]

    def __getitem__(self, idx):
        # add channel dimension
        out = (
            self.fbp[idx, ...].unsqueeze(0),
            self.sinogram[idx, ...].unsqueeze(0),
            self.phantom[idx, ...].unsqueeze(0),
        )
        # move to device and apply transformations
        if self.device is not None:
            out = tuple([x.to(self.device) for x in out])
        if self.transform is not None:
            out = self.transform(out)
        return out


def load_ct_data(subset, num_batches=3, **kwargs):
    """ Concatenates individual CTDatasets from four files.
    从四个目录下连接一个特定CT数据集

    参数：
    ----------
    subset : string
        'train', 'val', 或者 'test' 之一，或者有效的子目录路径
    **kwargs : dictionary
        传递给CT数据集的其他关键字参数。    

    返回：
    -------
    来自多个数据批处理文件的组合数据集。
    """

    if not subset == "val" and not subset == "test":
        num_batches = min(num_batches, 3)
    else:
        num_batches = 1

    return torch.utils.data.ConcatDataset(
        [
            CTDataset(subset, batch, **kwargs)
            for batch in range(1, num_batches + 1)
        ]
    )


# ---- run data exploration -----

if __name__ == "__main__":
    # validate data set and print some simple statistics
    tdata = load_ct_data("train", folds=10, num_fold=[0, 9], leave_out=True)
    vdata = load_ct_data("train", folds=10, num_fold=[0, 9], leave_out=False)
    print(len(tdata))
    print(len(vdata))
    y, z, x = tdata[0]
    print(y.shape, z.shape, x.shape)
    print(y.min(), z.min(), x.min())
    print(y.max(), z.max(), x.max())
