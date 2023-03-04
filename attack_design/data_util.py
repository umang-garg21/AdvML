from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import torch
from torchvision import transforms


class NormalizeByChannelMeanStd(torch.nn.Module):
    def __init__(self, mean, std):
        super(NormalizeByChannelMeanStd, self).__init__()
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean)
        if not isinstance(std, torch.Tensor):
            std = torch.tensor(std)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, tensor):
        return transforms.functional.normalize(tensor, self.mean, self.std)

    def extra_repr(self):
        return 'mean={}, std={}'.format(self.mean, self.std)


def cifar10_dataloader(batch_size=64, data_dir='./data/'):
    '''CIFAR10 dataloader for test set'''

    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    test_set = CIFAR10(data_dir, train=False, transform=test_transform, download=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # calculated from cifar10 dataset
    dataset_normalization = NormalizeByChannelMeanStd(
        mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
    
    return test_loader, dataset_normalization
