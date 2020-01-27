import torch
import torch.nn as nn
from torchvision import models,datasets,transforms
import numpy as np
import time
from utils_for_cloud import *
from models import VGG


def main():

    """
    This code sets up the data and loads the model obtained after ADMM based training
    for retraining to enforce pruning constraints.

    The function retrains_model present in the utils file enforces the hard sparsity 
    constraints on the weights on the model obtained after ADMM based training 
    and then retrains the model to improve the accuracy.

    """
    
    #model = LeNet5()
    model = VGG(n_class=10)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    Path = 'saved_model/admm_model/cifar10_vgg_acc_0.688'    # Path to the saved model after ADMM based training
    
    model.load_state_dict(torch.load(Path))

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
    num_epochs = 20
    log_step = 100

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-2)
    #optimizer = torch.optim.SGD(model.parameters(), lr =5e-4,momentum =0.9, weight_decay = 5e-4 )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones = [10], gamma = 0.1)

    
    
    #######  Re-Training ##############
    # Parameters
    prune_type = 'filter'
    
    # Number of non-zero filters at each convolutional layer
    l = {'conv1':32,'conv2':64,'conv3':128,'conv4':128,'conv5':256,'conv6':256,'conv7':256,'conv8':256}
    

    retrain_model(model,train_data,val_data,batch_size,loss_fn,num_epochs,log_step,optimizer,scheduler,l,prune_type,device)

    
    # Check the test accuracy
    model.eval()
    test_accuracy = eval_accuracy_data(test_data,model,batch_size,device)
    print('Test accuracy is',test_accuracy)
    
    





if __name__ == '__main__':
    main()

