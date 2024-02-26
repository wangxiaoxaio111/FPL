# -*- coding:utf-8 -*-
import os
import torch 
from sklearn.mixture import GaussianMixture
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
from data.cifar_aug_one import CIFAR10, CIFAR100
from data.mnist_aug_one import MNIST
from data.SVHN_aug_one import SVHN
from ResNet_cifar import resnet34
from model import CNN
import argparse, sys
import numpy as np
import datetime
import time
import shutil

# from loss import loss_coteaching

parser = argparse.ArgumentParser()
#parser.add_argument('--lr', type = float, default = 0.001)
parser.add_argument('--result_dir', type = str, help = 'dir to save result txt files', default = 'results/')
parser.add_argument('--noise_rate', type = float, help = 'corruption rate, should be less than 1', default = 0.5)
parser.add_argument('--forget_rate', type = float, help = 'forget rate', default = None)
parser.add_argument('--noise_type', type = str, help='[pairflip, symmetric,instance,tridiagonal]', default='instance')
parser.add_argument('--num_gradual', type = int, default = 10, help='how many epochs for linear drop rate, can be 5, 10, 15. This parameter is equal to Tk for R(T) in Co-teaching paper.')
parser.add_argument('--exponent', type = float, default = 1, help='exponent of the forget rate, can be 0.5, 1, 2. This parameter is equal to c in Tc for R(T) in Co-teaching paper.')
parser.add_argument('--top_bn', action='store_true')
parser.add_argument('--dataset', type = str, help = 'mnist, cifar10, or cifar100', default ='cifar10')
parser.add_argument('--n_epoch', type=int, default=200)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--print_freq', type=int, default=50)
parser.add_argument('--num_workers', type=int, default=4, help='how many subprocesses to use for data loading')
parser.add_argument('--num_iter_per_epoch', type=int, default=20000)
parser.add_argument('--epoch_decay_start', type=int, default=80)
parser.add_argument('--samp_num', default=50000, type=int)
# parser.add_argument('--samp_num', default=60000, type=int)
# parser.add_argument('--samp_num', default=73257, type=int)
parser.add_argument('--es', default=30, type=int)
parser.add_argument('--p_threshold', default=0.5, type=float, help='clean probability threshold')
parser.add_argument('--alpha', default=0.9, type=float)
parser.add_argument('--beta', default=0.09, type=float)
parser.add_argument('--gamma', default=0.1, type=float)

parser.add_argument('--lr', default=1e-3, type=float, help='learning_rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--batch_size', default=128, type=int, help='batch_size')
parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight_decay')
parser.add_argument('--num_time', default=1, type=int)
args = parser.parse_args()

print(args)

# Seed
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

# Hyper Parameters
batch_size = 128
learning_rate = args.lr 

# load dataset

if args.dataset == 'SVHN':
    input_channel = 3
    num_classes = 10
    args.top_bn = False
    args.epoch_decay_start = 80
    args.n_epoch = 200
    args.es = 30
    args.samp_num = 73257
    train_dataset = SVHN(root='/home/wangxiaoxiao/datasets/SVHN',
                         download=False,
                         train=True,
                         transform=transforms.ToTensor(),
                         noise_type=args.noise_type,
                         noise_rate=args.noise_rate
                         )

    test_dataset = SVHN(root='/home/wangxiaoxiao/datasets/SVHN',
                        download=False,
                        train=False,
                        transform=transforms.ToTensor(),
                        noise_type=args.noise_type,
                        noise_rate=args.noise_rate
                        )
if args.dataset=='mnist':
    input_channel=1
    num_classes=10
    args.top_bn = False
    args.epoch_decay_start = 80
    args.n_epoch = 200
    args.es = 12
    args.samp_num=60000
    train_dataset = MNIST(root='/home/wangxiaoxiao/datasets',
                                download=True,  
                                train=True, 
                                transform=transforms.ToTensor(),
                                noise_type=args.noise_type,
                                noise_rate=args.noise_rate
                         )
    
    test_dataset = MNIST(root='/home/wangxiaoxiao/datasets',
                               download=True,  
                               train=False, 
                               transform=transforms.ToTensor(),
                               noise_type=args.noise_type,
                               noise_rate=args.noise_rate
                        )
    
if args.dataset=='cifar10':
    input_channel=3
    num_classes=10
    args.top_bn = False
    args.epoch_decay_start = 80
    args.n_epoch = 200
    args.es=30
    args.samp_num = 50000
    train_dataset = CIFAR10(root='/home/wangxiaoxiao/datasets',
                                download=True,  
                                train=True, 
                                transform=transforms.ToTensor(),
                                noise_type=args.noise_type,
                                noise_rate=args.noise_rate
                           )
    
    test_dataset = CIFAR10(root='/home/wangxiaoxiao/datasets',
                                download=True,  
                                train=False, 
                                transform=transforms.ToTensor(),
                                noise_type=args.noise_type,
                                noise_rate=args.noise_rate
                          )

if args.dataset=='cifar100':
    input_channel=3
    num_classes=100
    args.top_bn = False
    args.epoch_decay_start = 100
    args.n_epoch = 200
    args.es = 25
    args.samp_num = 50000
    train_dataset = CIFAR100(root='/home/wangxiaoxiao/datasets',
                                download=True,  
                                train=True, 
                                transform=transforms.ToTensor(),
                                noise_type=args.noise_type,
                                noise_rate=args.noise_rate
                            )
    
    test_dataset = CIFAR100(root='/home/wangxiaoxiao/datasets',
                                download=True,  
                                train=False, 
                                transform=transforms.ToTensor(),
                                noise_type=args.noise_type,
                                noise_rate=args.noise_rate
                            )

if args.forget_rate is None:
    forget_rate=args.noise_rate
else:
    forget_rate=args.forget_rate

noise_or_not = train_dataset.noise_or_not

# Adjust learning rate and betas for Adam Optimizer
mom1 = 0.9
mom2 = 0.1
alpha_plan = [learning_rate] * args.n_epoch
beta1_plan = [mom1] * args.n_epoch
for i in range(args.epoch_decay_start, args.n_epoch):
    alpha_plan[i] = float(args.n_epoch - i) / (args.n_epoch - args.epoch_decay_start) * learning_rate
    beta1_plan[i] = mom2

def adjust_learning_rate(optimizer, epoch):
    for param_group in optimizer.param_groups:
        param_group['lr']=alpha_plan[epoch]
        param_group['betas']=(beta1_plan[epoch], 0.999) # Only change beta1
        
# define drop rate schedule
rate_schedule = np.ones(args.n_epoch)*forget_rate
rate_schedule[:args.num_gradual] = np.linspace(0, forget_rate**args.exponent, args.num_gradual)
   
save_dir = args.result_dir +'/' +args.dataset+'/coteaching/'

if not os.path.exists(save_dir):
    os.system('mkdir -p %s' % save_dir)

model_str=args.dataset+'_coteaching_'+args.noise_type+'_'+str(args.noise_rate)+'_'+str(args.num_time)

txtfile=save_dir+"/"+model_str+".txt"
nowTime=datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
if os.path.exists(txtfile):
    os.system('mv %s %s' % (txtfile, txtfile+".bak-%s" % nowTime))


def accuracy(logit, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    output = F.softmax(logit, dim=1)
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class SelfAdaptiveTrainingCE():
    def __init__(self, labels, num_classes=num_classes, momentum=0.9, es=21):
        # initialize soft labels to onthot vectors
        # self.feature_labels = torch.zeros(targets.shape[0], 10, dtype=torch.float).cuda(non_blocking=True)
        self.soft_labels = torch.zeros(labels.shape[0], num_classes, dtype=torch.float).cuda(non_blocking=True)
        self.soft_labels[torch.arange(labels.shape[0]), labels] = 1
        self.momentum = momentum
        self.es = es

    def __call__(self, logits, targets, index, epoch):
        if epoch < self.es:
            return F.cross_entropy(logits, targets)
        
        # obtain prob, then update running avg
        prob = F.softmax(logits.detach(), dim=1)
        self.soft_labels[index] = self.momentum * self.soft_labels[index] + (1 - self.momentum) * prob

        # obtain weights
        weights, _ = self.soft_labels[index].max(dim=1)
        weights *= logits.shape[0] / weights.sum()

        # compute cross entropy loss, without reduction
        loss = torch.sum(-F.log_softmax(logits, dim=1) * self.soft_labels[index], dim=1)

        # sample weighted mean
        #loss = (loss * weights).mean()
        return loss.mean()
    
    
# Train the Model
def train(train_loader,targets, soft_labels, epoch, model1, optimizer1, criterion):
    #print ('Training %s...' % model_str)
    pure_ratio_list=[]
    pure_ratio_1_list=[]
    pure_ratio_2_list=[]
    
    train_total=0
    train_correct=0 
    train_total2=0
    train_correct2=0
    feature_labels = torch.zeros(targets.shape[0], num_classes, dtype=torch.float).cuda(non_blocking=True)

    if epoch < args.es:
        print("WARM UP ING")

    else:
        print("MAKING SOFT LABEL")

        num_samples = args.samp_num
        all_targets = torch.zeros(args.samp_num)
        normalized_features = torch.zeros((num_samples, 128))
        predictions = torch.zeros(num_samples)

        with torch.no_grad():
            for i, (images, labels, indexes) in enumerate(train_loader):
                ind = indexes.cpu().numpy().transpose()
                if i > args.num_iter_per_epoch:
                    break
                images = Variable(images).cuda()
                labels = Variable(labels).cuda()
                # gt=Variable(gt).cuda()
                predict, feature = model1(images)
                normalized_fea = F.normalize(feature, dim=1)

                _, predicted = torch.max(predict, 1)

                for b in range(images.size(0)):
                    # all_targets[indexes[b]] = labels[b]
                    all_targets[indexes[b]] = np.argmax(soft_labels[indexes[b]].cpu(), axis=-1).cuda()
                    normalized_features[indexes[b]] = normalized_fea[b]
                    predictions[indexes[b]] = predicted[b]

        overall_distance = torch.zeros((num_samples,))
        all_prob = np.zeros((num_samples,))
        centers = torch.zeros((num_classes, 128))
        for cls_ in range(num_classes):
            centers[cls_] = F.normalize(normalized_features[predictions == cls_].mean(dim=0), dim=0)

        for cls_ in range(num_classes):
            distance_cls = (normalized_features[all_targets == cls_] * centers[cls_]).sum(dim=1)
            distance_cls = torch.where(torch.isnan(distance_cls), torch.full_like(distance_cls, 0), distance_cls)
            overall_distance[all_targets == cls_] = distance_cls

        overall_distance = (overall_distance - overall_distance.min()) / (
                overall_distance.max() - overall_distance.min())
        overall_distance = overall_distance.numpy().reshape(-1, 1)


        center_t = centers.permute(1, 0)

        overall_distance = torch.mm(normalized_features, center_t)

        norm_c = torch.sum(overall_distance, 0)


        f_label = np.argmax(overall_distance.cpu(), axis=-1)
        feature_labels[torch.arange(targets.shape[0]), f_label] = 1
        # feature_labels = F.softmax(feature_labels.detach())


    for i, (images, labels, indexes) in enumerate(train_loader):
        ind=indexes.cpu().numpy().transpose()
        if i>args.num_iter_per_epoch:
            break

        images = Variable(images).cuda()
        labels = Variable(labels).cuda()

        # Forward + Backward + Optimize
        logits1,_=model1(images)
        prec1, _ = accuracy(logits1, labels, topk=(1, 5))
        train_total+=1
        train_correct+=prec1

        if epoch < args.es:
            loss_1 = F.cross_entropy(logits1, labels)
            optimizer1.zero_grad()
            loss_1.backward()
            optimizer1.step()
        else:
            clean = []
            # for i in range(0, 128):
            #     if gmm_prob[indexes[i]] > args.p_threshold:
            #         clean.append(i)

            prob_1 = F.softmax(logits1.detach(), dim=1)
            # prob_2=F.one_hot(prob_1)
            soft_labels[ind] = args.alpha * soft_labels[ind] + args.beta * prob_1 + args.gamma * feature_labels[ind]
            
            # obtain weights
            weights, _ = soft_labels[indexes].max(dim=1)
            weights *= logits1.shape[0] / weights.sum()

            # compute cross entropy loss, without reduction
            loss = torch.sum(-F.log_softmax(logits1, dim=1) * soft_labels[indexes], dim=1)

            # sample weighted mean
            loss = (loss * weights).mean()

            optimizer1.zero_grad()
            loss.backward()
            optimizer1.step()
        # loss = criterion(logits1, labels, indexes, epoch)
        # optimizer1.zero_grad()
        # loss.backward()
        # optimizer1.step()

    train_acc1=float(train_correct)/float(train_total)

    return train_acc1,  pure_ratio_1_list

# Evaluate the Model
def evaluate(test_loader, model1):
    #print ('Evaluating %s...' % model_str)
    model1.eval()    # Change model to 'eval' mode.
    with torch.no_grad():
        correct1 = 0
        total1 = 0
        for images, labels, _ in test_loader:
            images = Variable(images).cuda()
            logits1,_ = model1(images)
            outputs1 = F.softmax(logits1, dim=1)
            _, pred1 = torch.max(outputs1.data, 1)
            total1 += labels.size(0)
            correct1 += (pred1.cpu() == labels).sum()

    
    acc1 = 100*float(correct1)/float(total1)
    return acc1


def main():
    # Data Loader (Input Pipeline)
    print ('loading dataset...')
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size, 
                                               num_workers=args.num_workers,
                                               drop_last=True,
                                               shuffle=True)
    
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size, 
                                              num_workers=args.num_workers,
                                              drop_last=True,
                                              shuffle=False)
                                              
                                              
    # soft_labels
    targets = np.asarray(train_dataset.train_labels.reshape(-1))
    soft_labels = torch.zeros(targets.shape[0], num_classes, dtype=torch.float).cuda(non_blocking=True)
    soft_labels[torch.arange(targets.shape[0]), targets] = 1
    
    # Define models
    print ('building model...')
    if args.dataset =='mnist':
        cnn1 = CNN(input_channel=input_channel, n_outputs=num_classes)
        cnn1.cuda()
    else:
        cnn1 = resnet34(num_classes=num_classes)
        cnn1.cuda()
    #print cnn1.parameters
    optimizer1 = torch.optim.Adam(cnn1.parameters(), lr=learning_rate)
    #optimizer1 = torch.optim.SGD(cnn1.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    criterion = SelfAdaptiveTrainingCE(labels=targets, num_classes=num_classes)

    mean_pure_ratio1=0

    save_dir = '/home/wangxiaoxiao/FPL/results/SAT' + '/'
    if not os.path.exists(save_dir):
        os.system('mkdir -p %s' % save_dir)
    model_str = args.dataset + '_' + args.noise_type + '_' + str(args.noise_rate)+'_'+str(args.num_time)
    txtfile = save_dir + "/" + model_str + ".txt"

    #save_root = '/home/wangzhen16b/Project/Co_teaching/Co_teaching_ori/model_saved/Cifar10_CE_'+str(args.noise_type)+str(args.noise_rate)+'_v11.pth'
    epoch=0
    train_acc1=0
    train_acc2=0
    best_test_acc1 = 0
    best_test_acc2 = 0
    # evaluate models with random weights
    test_acc1 =evaluate(test_loader, cnn1)
    print('Epoch [%d/%d] Test Accuracy: Model1 %.4f %% Pure Ratio1 %.4f %% ' % (epoch+1, args.n_epoch, test_acc1, mean_pure_ratio1))
    with open(txtfile, "a") as myfile:
        myfile.write(str(int(epoch)) + ' ' + str(train_acc1) + ' ' + str(test_acc1) + "\n")
    # save results
#    with open(txtfile, "a") as myfile:
#        myfile.write(str(int(epoch)) + ': '  + str(train_acc1) +' '  + str(train_acc2) +' '  + str(test_acc1) + " " + str(test_acc2) + ' '  + str(mean_pure_ratio1) + ' '  + str(mean_pure_ratio2) + "\n")
    Last10_1=[]

    # training
    for epoch in range(1, args.n_epoch):
        # train models
        cnn1.train()
        adjust_learning_rate(optimizer1, epoch)
        train_acc1, pure_ratio_1_list = train(train_loader, targets,soft_labels, epoch, cnn1, optimizer1, criterion)
        # evaluate models
        test_acc1 =evaluate(test_loader, cnn1)
        if test_acc1 > best_test_acc1:
            best_test_acc1 = test_acc1
            #torch.save(cnn1.state_dict(),save_root)

        # save results
        #mean_pure_ratio1 = sum(pure_ratio_1_list)/len(pure_ratio_1_list)
        print('Epoch [%d/%d] SAT Test Acc: Model1 %.4f %% , Best Test Acc: Model1 %.4f %%' % (epoch+1, args.n_epoch, test_acc1, best_test_acc1))
        with open(txtfile, "a") as myfile:
            myfile.write(str(int(epoch)) + ' ' + str(train_acc1) + ' ' + str(test_acc1) + "\n")

        if epoch>=args.n_epoch-10:
            Last10_1.append(test_acc1)

        if epoch==args.n_epoch-1:
            print('Last10_1:')
            print(Last10_1)
            print('mean±std: %.2f±%.2f' % (np.mean(Last10_1), np.std(Last10_1)))


#        with open(txtfile, "a") as myfile:
#            myfile.write(str(int(epoch)) + ': '  + str(train_acc1) +' '  + str(train_acc2) +' '  + str(test_acc1) + " " + str(test_acc2) + ' ' + str(mean_pure_ratio1) + ' ' + str(mean_pure_ratio2) + "\n")

if __name__=='__main__':
    t1 = time.localtime()
    print(t1)
    main()
    t2 = time.localtime()
    print(t2)
