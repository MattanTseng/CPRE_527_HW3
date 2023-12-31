import torch
import torchvision
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split


# based on code from TA
def load_cifar10(hyperparameters: dict):

    # pull out the ratio of images that are going to be used for training
    train_ratio =  hyperparameters["train_ratio"]

    # pull out the batch size that we want to use.
    batch_size = hyperparameters["batch_size"]
    
    # Define transformations for the images
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    
    all_cifar_data = torchvision.datasets.CIFAR10(root='./data', train=True,
                    download=True, transform=transform)
    
    
    # Download and load training dataset
    trainset, val_set = train_test_split(all_cifar_data, train_size=train_ratio, stratify=all_cifar_data.targets)
    
    # here's the validation set
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size,
                                            shuffle=True, num_workers=2)

    # here's the trainig set
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)

    

    # Download and load test dataset
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                            shuffle=False, num_workers=2)

    # changed this so that it's pulling from the CIFAR10 dataset instead of being hard coded
    class_names = ['plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    
    return train_loader,val_loader, test_loader,class_names, 