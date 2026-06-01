import torch
import torch.nn as nn
import torch.nn.functional as F
from .metrices import *

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True, smooth=1e10):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()                            
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)  
        return 1 - dice 

class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()
    
    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        BCE = F.binary_cross_entropy(inputs, targets, reduce='mean')
        return BCE

class DiceBCELoss(nn.Module):
    # this is also know as the combo loss
    def __init__(self, smooth=1e10):
        super(DiceBCELoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        BCE = F.binary_cross_entropy(inputs, targets, reduce='mean')
        intersection = (inputs * targets).sum()                            
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)  
        dice_BCE = dice+BCE
        return dice_BCE

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        # selecting alpha as 0.25 and gamma as seems to be the most common method
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        BCE = F.binary_cross_entropy(inputs, targets, reduce='mean')
        pt = torch.exp(-BCE)
        focal_loss = self.alpha*(1 - pt)**self.gamma*BCE
        return focal_loss

class LogCoshDiceLoss(nn.Module):
    # https://arxiv.org/pdf/2006.14822.pdf
    def __init__(self, smooth=1):
        super(LogCoshDiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()                            
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        LCDL = torch.log(torch.cosh(dice))
        return 1 - LCDL
    
class JaccardLoss(nn.Module):
    def __init__(self, smooth=1e10):
        super(JaccardLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        all = (inputs + targets).sum()
        union = all - intersection      # all includes all of A as well as B, to remove the intersecting points twince subtact
        IOU = (intersection + self.smooth)/(union + self.smooth)
        return 1-IOU    # to make IOU diff
    
class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, smooth=1e10):
        # selecting alpha=0.3, beta=0.7, gamma=0.75 as these seems to be the most common values
        # https://arxiv.org/pdf/1706.05721.pdf
        super(TverskyLoss, self).__init__()
        self.beta = beta
        self.alpha = 1-beta
        self.smooth = smooth
    
    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        tp = (inputs * targets).sum()
        fp = ((1-targets)*inputs).sum()
        fn = (targets*(1-inputs)).sum()
        tversky_index = (tp + self.smooth)/(tp + (self.beta * fp) + (self.alpha*fn) + self.smooth)
        return 1 - tversky_index

class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, gamma=1.25, smooth=1e10):
        # selecting alpha=0.7, beta=0.3, gamma=1.25 as these seems to be the most common values
        # https://arxiv.org/pdf/1810.07842.pdf
        super(FocalTverskyLoss, self).__init__()
        self.beta = beta
        self.alpha = 1-self.beta
        self.gamma = gamma
        self.smooth = smooth
    
    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        tp = (inputs * targets).sum()
        fp = ((1-targets)*inputs).sum()
        fn = (targets*(1-inputs)).sum()
        tversky_index = (tp + self.smooth)/(tp + (self.beta * fp) + (self.alpha*fn) + self.smooth)
        ftl = (1-tversky_index)**self.gamma
        return ftl

class UnifiedFocalLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, gamma=1.25,  delta=0.5 ,smooth=1e10):
        # https://www.sciencedirect.com/science/article/pii/S0895611121001750
        # selecting alpha=0.7, beta=0.3, gamma=1.25 as these seems to be the most common values
        super(UnifiedFocalLoss, self).__init__()
        self.beta = beta
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth
        self.delta = delta
    
    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        tp = (inputs * targets).sum()
        fp = ((1-targets)*inputs).sum()
        fn = (targets*(1-inputs)).sum()
        tversky_index = (tp + self.smooth)/(tp + (self.beta * fp) + (self.alpha*fn) + self.smooth)
        fl = 1-tversky_index
        ftl = (1-tversky_index)**self.gamma
        ufocal = self.delta*fl + (1-self.delta)*ftl
        return ufocal
    
class SensitivitySpecificityLoss(nn.Module):
    def __init__(self, alpha=0.3, smooth=1e10):
        # selecting alpha=0.3, beta=0.7, as these seems to be the most common values (specificity is higher for imbalalanced dataset)
        super(SensitivitySpecificityLoss, self).__init__()
        self.alpha = alpha
        self.beta = 1-self.alpha
        self.smooth = smooth
    
    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        tp = (inputs * targets).sum()
        tn = ((1-inputs)*(1-targets)).sum()
        fp = ((1-targets)*inputs).sum()
        fn = (targets*(1-inputs)).sum()
        sensitivity = tp + self.smooth / tp + fn + self.smooth
        specificity = tn + self.smooth / tn + fp + self.smooth
        ssl = self.alpha*sensitivity + self.beta * specificity
        return ssl
    
# error NAN
class ExponentialLogarithmicLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, gamma=0.3, smooth=1):
        # aplha, beta 0.5 and gamma 0.3 as per paper
        # https://arxiv.org/pdf/1809.00076.pdf
        # alpha is weight of dice, beta is weight of BCE
        super(ExponentialLogarithmicLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth
    
    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        BCE = F.binary_cross_entropy(inputs, targets, reduce='mean')
        intersection = (inputs * targets).sum()                            
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth) 
        BCE_exp = torch.exp(-(torch.log(BCE)**self.gamma))
        dice_exp = torch.exp(-(torch.log(1-dice)**self.gamma))
        
        # inpect for NaN
        print("BCE: ", BCE.item())
        print("Dice: ", dice.item())
        print("BCE Exp: ", BCE_exp.item())
        print("Dice Exp: ", dice_exp.item())

        ELL = self.alpha*dice_exp + self.beta*BCE_exp
        return ELL

class ShapeAwareLoss(nn.Module):
    pass

    
