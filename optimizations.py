import torch
from torch import Tensor

import numpy as np


class MSELoss_age_multiplied(torch.nn.MSELoss):
    def __init__(self, *args, **kwargs):
        super(MSELoss_age_multiplied, self).__init__(*args, **kwargs)
        self.multipliers = None

    def calibrate(self, dataset):
        vals = []
        min_age = dataset.age.min()
        max_age = dataset.age.max()
        counts = dataset.age.value_counts()
        min_count = counts.min()
        max_count = counts.max()

        for i in range(0, 100):
            if i < min_age or i > max_age:
                vals.append(int(max_count / min_count))
            else:
                vals.append(int(max_count / counts[i]))

        self.multipliers = vals

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        mse_tensor = super(MSELoss_age_multiplied, self).forward(input, target)
        l_target = [int(i) for i in target.flatten().tolist()]
        mul_list = [self.multipliers[i] for i in l_target]
        multiplied = Tensor(mul_list).to(device='cuda')
        return torch.mean(mse_tensor.mul(multiplied))

    # def backward(self, input: Tensor, target: Tensor) -> Tensor:
    #     mse_tensor = super(MSELoss_age_multiplied, self).backward(input, target)
    #     l_target = [int(i) for i in target.flatten().tolist()]
    #     mul_list = [self.multipliers[i] for i in l_target]
    #     multiplied = Tensor(mul_list).to(device='cuda')
    #     return mse_tensor.mul(multiplied)

