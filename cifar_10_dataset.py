from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as tt


def get_train_data(data_config):
    train_transform = tt.Compose([
        tt.RandomHorizontalFlip(p=0.5),
        tt.ToTensor(),
        tt.Normalize(mean=(0.5, ), std=(0.5, ))
    ])

    train_dataset = torchvision.datasets.CIFAR10(**data_config.dataset_config, 
                                            train=True, 
                                            transform=train_transform, 
                                            target_transform=None, 
                                            download=False)
    train_data_loader = DataLoader(dataset=train_dataset, 
                                   **data_config.data_loader_config, 
                                   shuffle=True)
    return train_dataset, train_data_loader
    
def get_val_data(data_config):
    test_transform = tt.Compose([
        tt.ToTensor(),
        tt.Normalize(mean=(0.5, ), std=(0.5, ))
    ])

    test_dataset = torchvision.datasets.CIFAR10(**data_config.dataset_config, 
                                            train=False, 
                                            transform=test_transform, 
                                            target_transform=None, 
                                            download=False)
    test_data_loader = DataLoader(dataset=test_dataset, 
                                   **data_config.data_loader_config, 
                                   shuffle=False)
    return test_dataset, test_data_loader