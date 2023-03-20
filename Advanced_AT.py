import argparse
import IPython
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from tqdm import tqdm

import data_util
import model_util
import numpy as np
import matplotlib.pyplot as plt
import time
from losses import cw_loss, noise_loss, trades_loss
from datasets import SemiSupervisedDataset, SemiSupervisedSampler, DATASETS


# ------------------------------- CUDA SETUP -----------------------------------
# should provide some improved performance
cudnn.benchmark = True
# useful setting for debugging
# cudnn.benchmark = False
# cudnn.deterministic = True

use_cuda = torch.cuda.is_available()
# ------------------------------------------------------------------------------

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
        '--model_path', default='resnet_cifar10.pth', help='Filepath to the trained model'
    )
    parser.add_argument(
        '--log_path', type=str, default='./log_file.txt'
    )
    parser.add_argument(
        '--loss_type', type=str, choices =['ce', 'cw', 'trades'], default='ce'
    )
    parser.add_argument(
        '--alpha', type=int, default=2, help="PGD Learning rate: alpha / 255"
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")

    # Dataset config
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=DATASETS,
                        help='The dataset to use for training)')
    parser.add_argument('--data_dir', default='data', type=str,
                        help='Directory where datasets are located')
    parser.add_argument('--svhn_extra', action='store_true', default=False,
                        help='Adds the extra SVHN data')

    # Generic training configs
    parser.add_argument('--seed', type=int, default=1,
                        help='Random seed. '
                            'Note: fixing the random seed does not give complete '
                            'reproducibility. See '
                            'https://pytorch.org/docs/stable/notes/randomness.html')
    parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                        help='Input batch size for training (default: 128)')
    parser.add_argument('--test_batch_size', type=int, default=500, metavar='N',
                        help='Input batch size for testing (default: 128)')
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='Number of epochs to train. '
                            'Note: we arbitrarily define an epoch as a pass '
                            'through 50K datapoints. This is convenient for '
                            'comparison with standard CIFAR-10 training '
                            'configurations.')
 
    # Eval config
    parser.add_argument('--eval_freq', default=1, type=int,
                        help='Eval frequency (in epochs)')
    parser.add_argument('--train_eval_batches', default=None, type=int,
                        help='Maximum number for batches in training set eval')
    parser.add_argument('--eval_attack_batches', default=1, type=int,
                        help='Number of eval batches to attack with PGD or certify '
                            'with randomized smoothing')

    # Adversarial / stability training config
    parser.add_argument('--loss', default='trades', type=str,
                        choices=('trades', 'noise'),
                        help='Which loss to use: TRADES-like KL regularization '
                            'or noise augmentation')
    parser.add_argument('--distance', '-d', default='l_2', type=str,
                        help='Metric for attack model: l_inf uses adversarial '
                            'training and l_2 uses stability training and '
                            'randomized smoothing certification',
                        choices=['l_inf', 'l_2'])
    parser.add_argument('--epsilon', default=0.031, type=float,
                        help='Adversarial perturbation size (takes the role of'
                            ' sigma for stability training)')
    parser.add_argument('--pgd_num_steps', default=10, type=int,
                        help='number of pgd steps in adversarial training')
    parser.add_argument('--pgd_step_size', default=0.007,
                        help='pgd steps size in adversarial training', type=float)
    parser.add_argument('--beta', default=6.0, type=float,
                        help='stability regularization, i.e., 1/lambda in TRADES')

    # Semi-supervised training configuration
    parser.add_argument('--aux_data_filename', default=None, type=str,
                        help='Path to pickle file containing unlabeled data and '
                            'pseudo-labels used for RST')
    parser.add_argument('--unsup_fraction', default=0.5, type=float,
                        help='Fraction of unlabeled examples in each batch; '
                            'implicitly sets the weight of unlabeled data in the '
                            'loss. If set to -1, batches are sampled from a '
                            'single pool')
    parser.add_argument('--aux_take_amount', default=None, type=int,
                        help='Number of random aux examples to retain. '
                            'None retains all aux data.')
    parser.add_argument('--remove_pseudo_labels', action='store_true',
                        default=False,
                        help='Performs training without pseudo-labels (rVAT)')
    parser.add_argument('--entropy_weight', type=float,
                        default=0.0, help='Weight on entropy loss')

    # Additional aggressive data augmentation
    parser.add_argument('--autoaugment', action='store_true', default=False,
                        help='Use autoaugment for data augmentation')
    parser.add_argument('--cutout', action='store_true', default=False,
                        help='Use cutout for data augmentation')




    args = parser.parse_args()
    return args

# --------------------------- DATA AUGMENTATION --------------------------------
def data_augment(args):
    
    if args.dataset == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
    elif args.dataset == 'svhn':
        # the WRN paper does no augmentation on SVHN
        # obviously flipping is a bad idea, and it makes some sense not to
        # crop because there are a lot of distractor digits in the edges of the
        # image
        transform_train = transforms.ToTensor()

    if args.autoaugment or args.cutout:
        assert (args.dataset == 'cifar10')
        transform_list = [
            transforms.RandomCrop(32, padding=4, fill=128),
            # fill parameter needs torchvision installed from source
            transforms.RandomHorizontalFlip()]
        if args.autoaugment:
            transform_list.append(CIFAR10Policy())
        transform_list.append(transforms.ToTensor())
        if args.cutout:
            transform_list.append(Cutout(n_holes=1, length=16))

        transform_train = transforms.Compose(transform_list)
        logger.info('Applying aggressive training augmentation: %s'
                    % transform_train)

    transform_test = transforms.Compose([
        transforms.ToTensor()])

    # ----------------- DATASET WITH AUX PSEUDO-LABELED DATA -----------------------
    trainset = SemiSupervisedDataset(base_dataset=args.dataset,
                                    add_svhn_extra=args.svhn_extra,
                                    root=args.data_dir, train=True,
                                    download=True, transform=transform_train,
                                    aux_data_filename=args.aux_data_filename,
                                    add_aux_labels= not args.remove_pseudo_labels,
                                    aux_take_amount=args.aux_take_amount)

    # num_batches=50000 enforces the definition of an "epoch" as passing through 50K

    # datapoints
    train_batch_sampler = SemiSupervisedSampler(
        trainset.sup_indices, trainset.unsup_indices,
        args.batch_size, args.unsup_fraction,
        num_batches=int(np.ceil(50000 / args.batch_size)))
    epoch_size = len(train_batch_sampler) * args.batch_size

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = DataLoader(trainset, batch_sampler = train_batch_sampler, **kwargs)

    testset = SemiSupervisedDataset(base_dataset=args.dataset,
                                    root=args.data_dir, train=False,
                                    download=True,
                                    transform=transform_test)
    test_loader = DataLoader(testset, batch_size=args.test_batch_size,
                            shuffle=False, **kwargs)

    trainset_eval = SemiSupervisedDataset(
        base_dataset=args.dataset,
        add_svhn_extra=args.svhn_extra,
        root=args.data_dir, train=True,
        download=True, transform=transform_train)

    eval_train_loader = DataLoader(trainset_eval, batch_size=args.test_batch_size,
                                shuffle=True, **kwargs)

    eval_test_loader = DataLoader(testset, batch_size=args.test_batch_size,
                                shuffle=False, **kwargs)
    # ------------------------------------------------------------------------------

    return trainset, trainset_eval, testset, train_loader, test_loader, eval_train_loader, eval_test_loader

def pgd_attack(model, x, y, eps, alpha, num_steps, loss_type, num_classes, args):
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
        elif loss_type == 'cw':
            loss = cw_loss(logits, y, loss_type, num_classes)
        elif loss_type == 'trades':
            optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
            (loss, natural_loss, robust_loss,
            entropy_loss_unlabeled) = trades_loss(
            model=model,
            x_natural=x,
            y=y,
            optimizer=optimizer,
            step_size= alpha,
            epsilon=eps,
            perturb_steps=num_steps,
            beta=args.beta,
            distance=args.distance,
            adversarial=args.distance == 'l_inf',
            entropy_weight=args.entropy_weight)

        loss.backward()
        grad = delta.grad.detach()

        delta.data = torch.clamp(delta + alpha * torch.sign(grad), -eps, eps)
        delta.grad.zero_()

    return x + delta.detach()

def adversarial_train(model, device, train_loader, val_loader, eps, alpha, num_steps, epochs, loss_type, num_classes, args):
    """
    Adversarial training using PGD attack
    """
    train_metrics = []
    #learning rate for model traning: 0.01, momentum: 0.05
    ckpt_pth = "Custom_training_model_200epochs.pt"
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
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
            # Use TRADES AS LOSS TYPE for adversarial example construction and loss.
            optimizer.zero_grad()

            # calculate robust loss
            if args.loss == 'trades':
                # The TRADES KL-robustness regularization term proposed by
                # Zhang et al., with some additional features
                (loss, natural_loss, robust_loss,
                entropy_loss_unlabeled) = trades_loss(
                    model=model,
                    x_natural=data,
                    y=target,
                    optimizer=optimizer,
                    step_size=alpha,
                    epsilon=eps,
                    perturb_steps=num_steps,
                    beta=args.beta,
                    distance=args.distance,
                    adversarial=args.distance == 'l_inf',
                    entropy_weight=args.entropy_weight)
            
            elif loss_type == 'noise':
                # Augmenting the input with random noise as in Cohen et al.
                assert (args.distance == 'l_2')
                loss = noise_loss(model=model, x_natural=data,
                                y=target, clamp_x=True, epsilon=epsilon)
                entropy_loss_unlabeled = torch.Tensor([0.])
                natural_loss = robust_loss = loss

            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\t Train Loss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))

            train_metrics.append(dict(
                epoch=epoch,
                loss=loss.item(),
                natural_loss=natural_loss.item(),
                robust_loss=robust_loss.item(),
                entropy_loss_unlabeled=entropy_loss_unlabeled.item()))

        model.eval()

        for batch_idx, (data, target) in enumerate(val_loader):
            
            loss_type = 'ce'
            data, target = data.to(device), target.to(device)  
            # PGD attack
            # target: target class (y)
            adv_data = pgd_attack(model, data, target, eps, alpha, num_steps, loss_type, num_classes, args)

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

            # print('target label values after epoch {}: {}'.format(epoch, target))
            # print('clean pred label values after epoch {}: {}'.format(epoch, pred_clean))
            # print('Adversarial pred label values after epoch {}: {}'.format(epoch, pred_adv))
            # print('Clean correct', correct_clean)
            # print('Adv Correct', correct_adv)
            
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
    plt.savefig('Accuracies vs epoch_Advanced_aT_After200epochs.png')

    np.savetxt("accuracy_clean_lst_custom_after_200epochs.csv", 
           accuracy_clean_lst,
           delimiter =", ", 
           fmt ='%f')

    np.savetxt("accuracy_adv_lst.csv_custom_after_200_epochs", 
           accuracy_adv_lst,
           delimiter =", ", 
           fmt ='%f')
          
def main():
    args = parse_args()
    device = args.device

    torch.cuda.current_device()
    # Load data
    train_loader, val_loader, test_loader, norm_layer = data_util.cifar10_dataloader(data_dir=args.data_dir)
    # trainset, trainset_eval, testset, train_loader, test_loader, eval_train_loader, eval_test_loader = data_augment(args)

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

    PATH = "Custom_training_model_200epochs.pt"
    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    # adversarial_train(model, args.device, train_loader, eval_train_loader, eps, alpha, num_steps, epochs, loss_type, num_classes, args)

    ############### TEST MODE ################

    ## Make sure the model is in `eval` mode.
    model.eval()
    correct_clean = 0
    correct_adv = 0
    
    for batch_idx, (data, target) in enumerate(test_loader):
        
        data, target = data.to(device), target.to(device)  
        # PGD attack
        # target: target class (y)
        adv_data = pgd_attack(model, data, target, eps, alpha, num_steps, loss_type, num_classes, args)
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