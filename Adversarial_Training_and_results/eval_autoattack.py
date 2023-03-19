import argparse
import IPython
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

import data_util
import model_util
import numpy as np
import matplotlib.pyplot as plt
import time

def parse_args():
    '''Parse input arguments'''
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eps", type=int, default=8, help="Attack budget: epsilon / 255"
    )
    parser.add_argument(
        '--norm', type=str, default='Linf', choices=['Linf', 'L2', 'L1'], help='Norm to use for attack'
    )
    parser.add_argument(
        '--data_dir', default='./data/', type=str, help="Folder to store downloaded dataset"
    )
    parser.add_argument(
        '--model_path', default='resnet_cifar10.pth', help='Filepath to the trained model'
    )
    parser.add_argument(
        '--batch_size', type=int, default=1024, help='Batch size for attack'
    )
    parser.add_argument(
        '--log_path', type=str, default='./log_file.txt'
    )
    parser.add_argument(
        '--loss_type', type=str, choices =['ce', 'cw'], default='ce'
    )
    parser.add_argument(
        '--alpha', type=int, default=2, help="PGD Learning rate: alpha / 255"
    )
    parser.add_argument(
        '--epochs', type=int, default=100, help="number of epochs for training"
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")
    
    args = parser.parse_args()
    return args

def cw_loss(logits, true_labels, targeted, num_classes):
    confidence_tensor = torch.zeros_like(true_labels)
    target_class = torch.ones_like(true_labels)
    if targeted:
        y_one_hot = F.one_hot(true_labels, num_classes)
        logits_y_one_hot = y_one_hot*logits
        logits_modified = logits - 1e8*y_one_hot
        vals, indices = torch.topk(logits_modified, 2, dim = -1)
        predict, second_best = torch.unbind(indices, dim= -1)

        x1 = torch.gather(logits, 1, predict.view(-1, 1))[:, 0]
        x2 = logits[:, true_labels]
            # Select the second best performing class and take logit difference
        loss = torch.max(torch.subtract(x1, x2), confidence_tensor)
    else:
        target_labels = torch.ones_like(true_labels)
        y_one_hot = F.one_hot(target_labels, num_classes)
        logits_y_one_hot = y_one_hot*logits
        logits_modified = logits - 1e8*y_one_hot

        vals, indices = torch.topk(logits_modified, 2, dim = -1)
        predict, second_best = torch.unbind(indices, dim= -1)
        
        x1 = torch.gather(logits, 1, predict.view(-1, 1))[:, 0]
        x2 = logits[:, target_labels]

        # Select the second best performing class and take logit difference
        loss = torch.max(torch.subtract(x1, x2), confidence_tensor)
    loss = loss.mean()
    return loss

def pgd_attack(model, x, y, eps, alpha, num_steps, loss_type, num_classes):
    """
    PGD attack on a model

    alpha: attack learning rate
    num_Steps: steps for PGD
    num_Classes: classes of training data
    eps: allowed perturbation ballpark
    loss_type: ce or cw
    """
    delta = torch.zeros_like(x).uniform_(-eps, eps)
    delta.requires_grad = True

    for t in range(num_steps):
        logits = model(x + delta)
        if loss_type == 'ce':
            loss = F.cross_entropy(logits, y)
        else:
            loss = cw_loss(logits, y, loss_type, num_classes)

        loss.backward()
        grad = delta.grad.detach()

        delta.data = torch.clamp(delta + alpha * torch.sign(grad), -eps, eps)
        delta.grad.zero_()

    return x + delta.detach()



def adversarial_train(model, device, train_loader, val_loader, eps, alpha, num_steps, epochs, loss_type, num_classes):
    """
    Adversarial training using PGD attack
    """
    #learning rate for model traning: 0.01, momentum: 0.05
    ckpt_pth = "adv_trained_model_200epochs_new.pt"
    optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.5)
    loss_lst = []
    accuracy_adv_lst = []
    accuracy_clean_lst = []
    epoch_lst = [i+1 for i in range(epochs)]
    best_adv_accuracy = 0

    model.train()
    for epoch in range(epochs): 
        correct = 0
        correct_clean = 0
        correct_adv= 0
        for batch_idx, (data, target) in enumerate(train_loader):

            data, target = data.to(device), target.to(device)  
            # PGD attack
            # target: target class (y)
            adv_data = pgd_attack(model, data, target, eps, alpha, num_steps, loss_type, num_classes)

            # Train on adversarial examples
            optimizer.zero_grad()
            output = model(adv_data)
            pred = torch.argmax(torch.softmax(output, dim=1), dim=1)

            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\t Train Loss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))

        model.eval()

        for batch_idx, (data, target) in enumerate(val_loader):
            
            data, target = data.to(device), target.to(device)  
            # PGD attack
            # target: target class (y)
            adv_data = pgd_attack(model, data, target, eps, alpha, num_steps, loss_type, num_classes)

            # Train on adversarial examples
            optimizer.zero_grad()
            output_clean  = model(data)
            output_adv = model(adv_data)

            #print("output clean shape", output_clean.shape)
            #print("output clean shape", output_adv.shape)
            pred_clean = torch.argmax(torch.softmax(output_clean, dim=1), dim=1)  
            pred_adv = torch.argmax(torch.softmax(output_adv, dim=1), dim=1)  

            loss_clean = F.cross_entropy(output_clean, target)
            loss_adv = F.cross_entropy(output_adv, target)

            if batch_idx % 100 == 0:
                print('Epoch: {} [{}/{} ({:.0f}%)]\t Validation Loss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(val_loader.dataset),
                    100. * batch_idx / len(val_loader), loss.item()))

            correct_clean += (pred_clean.cpu() == target.cpu()).float().sum()
            correct_adv += (pred_adv.cpu() == target.cpu()).float().sum()

            #print('target label values after epoch {}: {}'.format(epoch, target))
            #print('clean pred label values after epoch {}: {}'.format(epoch, pred_clean))
            #print('Adversarial pred label values after epoch {}: {}'.format(epoch, pred_adv))
            #print('Clean correct', correct_clean)
            #print('Adv Correct', correct_adv)
        accuracy_clean = 100 * correct_clean / len(val_loader.dataset)
        accuracy_adv = 100 * correct_adv / len(val_loader.dataset)
        print("Clean Validation Accuracy % = {}".format(accuracy_clean))  
        print("Robust Validation Accuracy % = {}".format(accuracy_adv))

        accuracy_clean_lst.append(accuracy_clean)
        accuracy_adv_lst.append(accuracy_adv)

        if accuracy_adv >= best_adv_accuracy:
            torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                    }, ckpt_pth)

    plt.figure()
    # plt.plot(epoch_lst, loss_lst, label='learning curve')
    plt.plot(epoch_lst, accuracy_clean_lst, label='Clean accuracy')
    plt.plot(epoch_lst, accuracy_adv_lst, label='Robust Accuracy')
    plt.xlabel('epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('Accuracies vs epoch.png')

    np.savetxt("accuracy_clean_lst.csv", 
           accuracy_clean_lst,
           delimiter =", ", 
           fmt ='%f')

    np.savetxt("accuracy_adv_lst.csv", 
           accuracy_adv_lst,
           delimiter =", ", 
           fmt ='%f')
          
def main():
    args = parse_args()
    device = args.device

    # Load data
    train_loader, val_loader, test_loader, norm_layer = data_util.cifar10_dataloader(data_dir=args.data_dir)
    model = model_util.ResNet18(num_classes=10)
    model.normalize = norm_layer
    model.load(args.model_path, args.device)
    model = model.to(args.device)

    eps = args.eps / 255
    alpha = args.alpha / 255
    num_steps = 50 # PGD attack steps
    epochs = args.epochs
    loss_type = args.loss_type
    device = args.device
    num_classes = 10 # number of classes = 10
    
    # Train the model
    # adversarial_train(model, args.device, train_loader, val_loader, eps, alpha, num_steps, epochs, loss_type, num_classes)

    ############### TEST MODE ################
    PATH = '/home/teamteam/AdvML/HW2/adv_trained_model_200epochs.pt'
    # checkpoint = torch.load(PATH, map_location='cuda')
    # model.load_state_dict(checkpoint['model_state_dict'])

    ## Make sure the model is in `eval` mode.
    model.eval()
    correct_clean = 0
    correct_adv = 0
    
    for batch_idx, (data, target) in enumerate(test_loader):
        
        data, target = data.to(device), target.to(device)  
        # PGD attack
        # target: target class (y)
        adv_data = pgd_attack(model, data, target, eps, alpha, num_steps, loss_type, num_classes)
        output_clean  = model(data)
        output_adv = model(adv_data)

        pred_clean = torch.argmax(torch.softmax(output_clean, dim=1), dim=1)  
        pred_adv = torch.argmax(torch.softmax(output_adv, dim=1), dim=1)  
        loss_clean = F.cross_entropy(output_clean, target)
        loss_adv = F.cross_entropy(output_adv, target)
        correct_clean += (pred_clean.cpu() == target.cpu()).float().sum()
        correct_adv += (pred_adv.cpu() == target.cpu()).float().sum()

        #print('target label values after epoch {}: {}'.format(epoch, target))
        #print('clean pred label values after epoch {}: {}'.format(epoch, pred_clean))
        #print('Adversarial pred label values after epoch {}: {}'.format(epoch, pred_adv))
        #print('Clean correct', correct_clean)
        #print('Adv Correct', correct_adv)

    accuracy_clean = 100 * correct_clean / len(test_loader.dataset)
    accuracy_adv = 100 * correct_adv / len(test_loader.dataset)

    print("Clean Test Accuracy % = {}".format(accuracy_clean))  
    print("Robust Test Accuracy % = {}".format(accuracy_adv))

    # load attack 
    from autoattack import AutoAttack
    adversary = AutoAttack(model, norm=args.norm, eps=eps, log_path=args.log_path,
        version='standard', device=args.device)
    
    l = [x for (x, y) in test_loader]
    x_test = torch.cat(l, 0)
    l = [y for (x, y) in test_loader]
    y_test = torch.cat(l, 0)
    
    adv_complete = adversary.run_standard_evaluation(x_test, y_test, bs=args.batch_size)


if __name__ == "__main__":
    main()