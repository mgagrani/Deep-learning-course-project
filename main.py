import torch
import torch.nn as nn
from torchvision import models,datasets,transforms
import numpy as np
import time
from utils_for_cloud import train,eval_accuracy,eval_accuracy_data
from models import VGG



def main():

    """
    This code implements the conventional training pipeline for a CNN for obtaining the baseline 
    model for the ADMM based training
    
    """

    #model = LeNet5()
    model = VGG(n_class=10)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    #Path = 'saved_model/cifar10_vgg_acc_0.939'    
    #model.load_state_dict(torch.load(Path))

    model.to(device)

    #data_transforms = transforms.Compose([transforms.CenterCrop(32),transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
    train_data = datasets.CIFAR10('data/', train=True, download=False,
                         transform=transforms.Compose([
                             transforms.Pad(4),
                             transforms.RandomCrop(32),
                             transforms.RandomHorizontalFlip(),
                             transforms.ToTensor(),
                             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                         ]))
    #train_data = datasets.MNIST(root='data/',download=False,train=True,transform=data_transforms)

    # Splitting the training dataset into training and validation dataset
    N_train = len(train_data)
    val_split = 0.1
    N_val = int(val_split*N_train)

    train_data,val_data = torch.utils.data.random_split(train_data,(N_train-N_val,N_val))
    

    ## Test data
    test_data = datasets.CIFAR10('data/', train=False, download=False,
                         transform=transforms.Compose([
                             transforms.Pad(4),
                             transforms.RandomCrop(32),
                             transforms.RandomHorizontalFlip(),
                             transforms.ToTensor(),
                             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                         ]))
    #test_data = datasets.MNIST(root='data/',download=False,train=False,transform=data_transforms)

    batch_size = 128
    num_epochs = 50
    log_step = 100

    loss_fn = nn.CrossEntropyLoss()
    #optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)
    optimizer = torch.optim.SGD(model.parameters(), lr =5e-4,momentum =0.9, weight_decay = 5e-4 )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones = [15,30], gamma = 0.1)

    train(model,train_data,val_data,batch_size,loss_fn,optimizer,scheduler,num_epochs,log_step,device,save_model=True) 

    #test_sample = iter(torch.utils.data.DataLoader(test_data,batch_size=len(test_data))).next()
    #X_test,y_test = test_sample[0].to(device),test_sample[1].to(device)
    print('Test accuracy is',eval_accuracy_data(test_data,model,batch_size,device))
    torch.save(model)





if __name__ == '__main__':
    main()

        