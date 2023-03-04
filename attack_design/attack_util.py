import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

### Do not modif the following codes
class ctx_noparamgrad(object):
    def __init__(self, module):
        self.prev_grad_state = get_param_grad_state(module)
        self.module = module
        set_param_grad_off(module)

    def __enter__(self):
        pass

    def __exit__(self, *args):
        set_param_grad_state(self.module, self.prev_grad_state)
        return False
        
def get_param_grad_state(module):
    return {param: param.requires_grad for param in module.parameters()}

def set_param_grad_off(module):
    for param in module.parameters():
        param.requires_grad = False

def set_param_grad_state(module, grad_state):
    for param in module.parameters():
        param.requires_grad = grad_state[param]
### Ends


### PGD Attack
class PGDAttack():
    def __init__(self, attack_step=10, eps=8 / 255, alpha=2 / 255, loss_type='ce', targeted=False, 
                 num_classes=10):
        '''
        attack_step: number of PGD iterations
        eps: attack budget
        alpha: PGD attack step size
        '''
        self.eps = eps
        self.attack_step = attack_step
        self.alpha = alpha
        self.num_classes = num_classes
        self.loss_type = loss_type
        self.targeted = targeted
        self.norm = 'inf'

    def ce_loss(self, logits, y):
         loss_fn = nn.CrossEntropyLoss()
         loss = loss_fn(logits, y) 
         return loss
        
    def cw_loss(self, logits, true_labels):
        confidence_tensor = torch.zeros_like(true_labels)
        target_class = torch.ones_like(true_labels)
        if not self.targeted:
            y_one_hot = F.one_hot(true_labels, self.num_classes)
            logits_y_one_hot = y_one_hot*logits
            logits_modified = logits - 1e8*y_one_hot

            # print(logits_y_one_hot)
            # print(logits_modified)
            vals, indices = torch.topk(logits_modified, 2, dim = -1)
            predict, second_best = torch.unbind(indices, dim= -1)
            # print(true_labels)
            #print(predict)
            # print(logits)
            # print(torch.gather(logits, 1, true_labels.view(-1, 1)))
            # print(torch.gather(logits, 1, predict.view(-1, 1)))
            x1 = torch.gather(logits, 1, predict.view(-1, 1))[:, 0]
            x2 = logits[:, true_labels]
                # Select the second best performing class and take logit difference
            loss = torch.max(torch.subtract(x1, x2), confidence_tensor)
        else:
            target_labels = torch.ones_like(true_labels)
            y_one_hot = F.one_hot(target_labels, self.num_classes)
            logits_y_one_hot = y_one_hot*logits
            logits_modified = logits - 1e8*y_one_hot

            # print(logits_y_one_hot)
            # print(logits_modified)
            vals, indices = torch.topk(logits_modified, 2, dim = -1)
            predict, second_best = torch.unbind(indices, dim= -1)
            # print(true_labels)
            #print(predict)
            # print(logits)
            # print(torch.gather(logits, 1, true_labels.view(-1, 1)))
            # print(torch.gather(logits, 1, predict.view(-1, 1)))
            x1 = torch.gather(logits, 1, predict.view(-1, 1))[:, 0]
            x2 = logits[:, target_labels]
                # Select the second best performing class and take logit difference
            loss = torch.max(torch.subtract(x1, x2), confidence_tensor)
        loss = loss.mean()
        print(loss)
        return loss

    def perturb(self, model: nn.Module, X, y):
        delta = torch.zeros_like(X)
        gradients = torch.zeros_like(X)
        
        #Initial copy
        X_adv = X.clone().detach().requires_grad_(True).to(X.device)
        for it in range(self.attack_step):
            
            # Copy to temporary tensor to retain the grad values of X_adv after iterating
            X_adv_temp = X_adv.clone().detach().requires_grad_(True)
            logits = model.forward(X_adv_temp)
            if self.loss_type == 'cw':
                probs = F.softmax(logits, dim=1)
                vals, indices = torch.topk(logits, 2, dim = -1)
                attack_loss = self.cw_loss(logits, y)  # using label 1 for the targeted attack
            else:
                attack_loss = self.ce_loss(logits, y)  
            attack_loss.backward()   
            temp = attack_loss.cpu().detach().numpy()
            # print(temp)
            
            with torch.no_grad():
                if self.norm == 'inf':
                    gradients=  self.alpha * X_adv_temp.grad.sign()

                if self.targeted:
                    # Targeted: Gradient descent with on the loss of the (incorrect) target label
                    # w.r.t. the image data
                    X_adv -= gradients
                else:
                    # Untargeted: Gradient ascent on the loss of the correct label w.r.t.
                    # the model parameters
                    X_adv += gradients

            # L'inf' projection
            # Clip by minimizing on the positive side and maximizing on the negative side
            
            X_adv = torch.maximum(torch.minimum(X_adv, X+ self.eps), X - self.eps)
            
        delta = X_adv - X
        return delta

### FGSMAttack
'''
Technically you can transform your PGDAttack to FGSM Attack by controling parameters like `attack_step`. 
If you do that, you do not need to implement FGSM in this class.
'''
class FGSMAttack():
    def __init__(self, eps=8 / 255, loss_type='ce', targeted=True, num_classes=10):
        pass

    def perturb(self, model: nn.Module, X, y):
        delta = torch.ones_like(X)
        ### Your code here

        ### Your code ends
        return delta
