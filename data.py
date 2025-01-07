import torch 
from torch import nn 

from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import os


class corruptMnistDataset(Dataset):
    def __init__(self, images, labels):
        super().__init__()
        self.images = images
        self.labels = labels
        
    def __getitem__(self, index):
        image_tensor = self.images[index]
        label = self.labels[index]
        
        return image_tensor, label
    
    def __len__(self):
        return len(self.images)

def corrupt_mnist(batchsize= 64):
     
    dir = "./corruptmnist_v1"
    train_images = torch.stack([torch.load(dir+f"/train_images_{x}.pt") for x in range(0,6)], dim =0).flatten(end_dim=1)
    train_targets =torch.stack([torch.load(dir+f"/train_target_{x}.pt") for x in range(0,6)], dim=0).flatten(end_dim=1)
    
    test_images = torch.load(dir+"/test_images.pt")
    test_targets = torch.load(dir+"/test_target.pt")
    
    train_set = corruptMnistDataset(train_images,train_targets)
    test_set = corruptMnistDataset(test_images,test_targets)
    
    trainLoader = DataLoader(train_set,batch_size = 64, shuffle= True)
    testLoader = DataLoader(test_set,batch_size = 64, shuffle= False)
    
    return trainLoader, testLoader
    

corrupt_mnist()