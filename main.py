import os
import torch
import argparse
import torch.nn as nn
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
from timm import create_model
from fastai.data.core import Transform,TfmdLists,DataLoaders
from fastai.vision.core import PILImage,TensorImage,ToTensor,IntToFloatTensor,imagenet_stats
import re
import numpy as np
from fastai.learner import Learner
from fastai.metrics import accuracy
import torchvision.transforms as transforms
from PIL import Image
from fastai.vision.augment import Resize, Rotate, Flip, Zoom, Warp, Brightness, Contrast,Normalize
from fastai.vision.learner import create_head,has_pool_type
from torch.nn import Module
from torch import Tensor
from fastai.optimizer import Adam
from fastai.callback.schedule import *
import torch.nn.functional as F
from sklearn.metrics import roc_curve, auc
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd

def num_features_model(model):
    children = list(model.children())
    if len(children)==0: return None
    if isinstance(children[-1], nn.Sequential):
        return find_num_features_model(children[-1])
    else:
        for i in range(len(children)-1, -1, -1):
            if hasattr(children[i], 'num_features'): return children[i].num_features
    return None

class CustomTransform(Transform):
    def __init__(self, files, labels, is_valid=False):
        self.files = files
        self.labels = labels
        self.is_valid = is_valid
        self.label_to_int = {'bin': 1, 'sin': 0}

    def encodes(self, i):
        img = PILImage.create(self.files[i]).convert("RGB")
        img = np.array(img)    
        img = img.transpose((2, 0, 1)) 
        label = self.labels[i]   
        return (TensorImage(img), self.label_to_int[label])

class LabelSmoothing_CrossEntropy(Module):
    def __init__(self, eps:float=0.1, weight:Tensor=None, reduction:str='mean'): 
        super().__init__()
        self.eps = eps
        self.weight = weight
        self.reduction = reduction
        self.y_int = True 

    def forward(self, output:Tensor, target:Tensor) -> Tensor:
        c = output.size()[1]
        log_preds = F.log_softmax(output, dim=1)
        if self.reduction=='sum': loss = -log_preds.sum()
        else:
            loss = -log_preds.sum(dim=1) 
            if self.reduction=='mean':  loss = loss.mean()
        return loss*self.eps/c + (1-self.eps) * F.nll_loss(log_preds, target.long(), weight=self.weight, reduction=self.reduction)

    def activation(self, out:Tensor) -> Tensor: 
        return F.softmax(out, dim=-1)
    
    def decodes(self, out:Tensor) -> Tensor:
        return out.argmax(dim=-1)


class Pipeline:
    def __init__(self, data_path, bs, num_workers):
        self.data_path = data_path
        self.bs = bs
        self.num_workers = num_workers
        self.files, self.labels = self.get_files_labels()
        self.loss_func = LabelSmoothing_CrossEntropy(eps=0.1)

    def get_files_labels(self):
        files = []
        labels = []
        for label in os.listdir(self.data_path):
            for file in os.listdir(f'{self.data_path}/{label}'):
                files.append(f'{self.data_path}/{label}/{file}')
                labels.append(label)
        return files, labels

    def split_data(self):
        idxs = np.random.permutation(range(len(self.files)))
        cut = int(0.8 * len(self.files))
        self.train_files = [self.files[i] for i in idxs[:cut]]
        self.valid_files = [self.files[i] for i in idxs[cut:]]
        self.train_labels = [self.labels[i] for i in idxs[:cut]]
        self.valid_labels = [self.labels[i] for i in idxs[cut:]]


    def build_dataloaders(self):
        train_tl = TfmdLists(range(len(self.train_files)), CustomTransform(self.train_files, self.train_labels))
        valid_tl = TfmdLists(range(len(self.valid_files)), CustomTransform(self.valid_files, self.valid_labels, is_valid=True))

        aug_transforms = [Resize(448), Rotate(max_deg=30), Flip(), Zoom(min_zoom=1.0, max_zoom=1.5), 
                          Warp(), Brightness()]

        self.dataloaders = DataLoaders.from_dsets(train_tl, valid_tl, 
                                                  after_item=[Resize(448), ToTensor], 
                                                  after_batch=[*aug_transforms, IntToFloatTensor, Normalize.from_stats(*imagenet_stats)],
                                                  bs=self.bs, num_workers=self.num_workers)
        self.dataloaders = self.dataloaders.cuda()

    def create_timm_body(self, arch='efficientnet_b3a', pretrained=True, cut=None):
        model = create_model(arch, pretrained=pretrained)
        if cut is None:
            ll = list(enumerate(model.children()))
            cut = next(i for i, o in reversed(ll) if has_pool_type(o))
        if isinstance(cut, int): 
            return nn.Sequential(*list(model.children())[:cut])
        elif callable(cut): 
            return cut(model)
        else: 
            raise ValueError("cut must be either integer or function")

    def create_head(self, nf, num_classes):
        return create_head(nf, num_classes, first_bn=True, bn_final=True, lin_first=False)

    def init_model(self, m, init_func):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            init_func(m.weight)
            if m.bias is not None: 
                m.bias.data.zero_()

    def build_model(self):
        num_classes = len(set(CustomTransform(self.train_files, self.train_labels).labels))
        body = self.create_timm_body('efficientnet_b3a', pretrained=True)
        nf = num_features_model(nn.Sequential(*body.children())) # Define the num_features_model function outside the class
        head = self.create_head(nf, num_classes)
        model = nn.Sequential(body, head)
        self.init_model(model[1], nn.init.kaiming_normal_)
        self.model = model
        return model

    def find_learning_rate(self):
        suggested_lrs = self.learn.lr_find()
        return suggested_lrs.valley

    def train_model(self, freeze_epochs=5, unfreeze_epochs=25):
        self.learn = Learner(self.dataloaders, self.model, loss_func=self.loss_func, opt_func=Adam, metrics=accuracy)
        self.learn.model = nn.DataParallel(self.learn.model, device_ids=[0, 1, 2, 3])
        self.learn.freeze()
        freeze_lr = self.find_learning_rate()
        self.learn.fit_one_cycle(freeze_epochs, lr_max=freeze_lr)
        self.learn.unfreeze()
        unfreeze_lr = self.find_learning_rate()
        self.learn.fit_one_cycle(unfreeze_epochs, lr_max=unfreeze_lr)    
        self.learn.save("best_model")
        
        
        interp = ClassificationInterpretation.from_learner(self.learn)
        conf_mat = interp.confusion_matrix()
        conf_mat_list = conf_mat.tolist()
        
        probs, targets = self.learn.get_preds()
        fpr, tpr, thresholds = roc_curve(targets, probs[:, 1])
        roc_auc = auc(fpr, tpr)
        return fpr, tpr, roc_auc, conf_mat_list
    
    def plot_roc_confusion_matrix(self, fpr, tpr, roc_auc, conf_mat_list):
        df_cm = pd.DataFrame(conf_mat_list, ['Binary', 'Single star'], ['Binary', 'Single star'])
        cmap = sns.cubehelix_palette(as_cmap=True)
        lw = 2
        plt.figure(figsize=(5, 10))
        plt.subplot(2,1,1)
        plt.plot(fpr, tpr, color='#D582A4',
                 lw=lw, label='ROC curve (area = %0.3f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color="darkorange", lw=lw, linestyle=':')
        plt.xlim([-0.01, 1])
        plt.ylim([0.0, 1.01])
        plt.xlabel('False Positive Rate', fontsize=13)
        plt.ylabel('True Positive Rate', fontsize=13)
        plt.title('ROC curve', fontsize=15)
        plt.legend(loc="lower right")
        
        plt.subplot(2,1,2)
        sns.heatmap(df_cm, square=True,cmap=cmap,annot=True, annot_kws={"size": 16},fmt='g') # font size
        plt.title("Confusion Matrix", fontsize=15)
        plt.xlabel("Predicted", fontsize=13)
        plt.ylabel("Actual", fontsize=13)
        plt.savefig('roc-cm.pdf', format="pdf", dpi=500,bbox_inches ='tight')
        
    def run_pipeline(self):
        self.get_files_labels()
        self.split_data()
        self.build_dataloaders()
        self.build_model()
        fpr, tpr, roc_auc, conf_mat_list = self.train_model()
        self.plot_roc_confusion_matrix(fpr, tpr, roc_auc, conf_mat_list)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the image classification pipeline")
    parser.add_argument('--data_path', type=str, required=True, help='Path to the data')
    parser.add_argument('--bs', type=int, default=64, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
    args = parser.parse_args()
    pipeline = Pipeline(data_path=args.data_path, bs=args.bs, num_workers=args.num_workers)
    pipeline.run_pipeline()
    
    