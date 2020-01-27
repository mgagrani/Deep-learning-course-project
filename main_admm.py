import torch
import torch.nn as nn
from torchvision import models,datasets,transforms
import numpy as np
import time
from utils_for_cloud import train_model_admm,Projection_structured,Projection_unstructured,eval_accuracy_data
from models import VGG
import os




def main():

    """
    This code implements the ADMM based training of a CNN. 
    """


    #model = LeNet5()
    model = VGG(n_class=10)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    Path = 'saved_model/pre_train_models/cifar10_vgg_acc_0.943'  # Path to the baseline model
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

    
    """
    N_train = len(train_data)
    val_split = 0.1
    N_val = int(val_split*N_train)

    train_data,val_data = torch.utils.data.random_split(train_data,(N_train-N_val,N_val))
    """

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
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-2)
    #optimizer = torch.optim.SGD(model.parameters(), lr =5e-4,momentum =0.9, weight_decay = 5e-4 )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones = [15,30], gamma = 0.1)

    
    
    ####### ADMM Training ##############
    ## Parameters
    
    fc_prune = False            # True if the fully connected layers are also pruned
    prune_type = 'filter'       # Type of structural pruning at the convolutional layers
    
    # Number of non zero filters at each convolutional layer
    l = {'conv1':32,'conv2':64,'conv3':128,'conv4':128,'conv5':256,'conv6':256,'conv7':256,'conv8':256}

    # ADMM parameters
    rho_val = 1.5e-3
    num_admm_steps = 10
    


    Z = {}
    U = {}
    rho = {}
    best_accuracy = 0
    all_acc = False
    
    ## Initialization of the variable Z and dual variable U
    
    for name_net in model.named_modules():
        name,net = name_net
        
        if isinstance(net,nn.Conv2d):
            Z[name] = net.weight.clone().detach().requires_grad_(False)
            Z[name] = Projection_structured(Z[name],l[name],prune_type)
            U[name] = torch.zeros_like(net.weight,requires_grad=False)
            rho[name] = rho_val

        
        elif fc_prune and isinstance(net,nn.Linear):
            Z[name] = net.weight.clone().detach().requires_grad_(False)
            l_unst = int(len(net.weight.data.reshape(-1,))*prune_ratio)
            Z[name],_ = Projection_unstructured(Z[name],l_unst)
            U[name] = torch.zeros_like(net.weight,requires_grad=False)
            
    
    ## ADMM loop
    
    for i in range(num_admm_steps):
        print('ADMM step number {}'.format(i))
        # First train the VGG model 
        train_model_admm(model,train_data,batch_size,loss_fn,optimizer,scheduler,num_epochs,log_step,Z,U,rho,fc_prune,device)
        
        # Update the variable Z
        for name_net in model.named_modules():
            name,net = name_net
            if isinstance(net,nn.Conv2d):
                Z[name] = Projection_structured(net.weight.detach()+U[name],l[name],prune_type)
            
            elif fc_prune and isinstance(net,nn.Linear):
                l_unst = int(len(net.weight.data.reshape(-1,))*prune_ratio)
                Z[name],_ = Projection_unstructured(net.weight.detach() + U[name],l_unst)
        
        # Updating the dual variable U
        for name_net in model.named_modules():
            name,net = name_net
            if isinstance(net,nn.Conv2d):
                U[name] = U[name] + net.weight.detach() - Z[name]
            elif fc_prune and isinstance(net,nn.Linear):
                U[name] = U[name] + net.weight.detach() - Z[name]
        
        
        ## Check the test accuracy
        model.eval()
        test_accuracy = eval_accuracy_data(test_data,model,batch_size,device)
        print('Test accuracy is',test_accuracy)
        if test_accuracy>best_accuracy:
            print('Saving model with test accuracy {:.3f}'.format(test_accuracy))
            torch.save(model.state_dict(),'saved_model/admm_model/cifar10_vgg_acc_{:.3f}'.format(test_accuracy))
            if all_acc:
                print('Removing model with test accuracy {:.3f}'.format(best_accuracy))
                os.remove('saved_model/admm_model/cifar10_vgg_acc_{:.3f}'.format(best_accuracy))
            best_accuracy = test_accuracy
            all_acc = True
    
    
    





if __name__ == '__main__':
    main()

        