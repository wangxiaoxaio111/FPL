# -*- coding:utf-8 -*-
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.mixture import GaussianMixture
from torch.autograd import Variable
import torchvision.transforms as transforms
from sklearn.cluster import KMeans
from data.FashionMNIST import Fashion
from data.SVHN import SVHN
from data.cifar import CIFAR10, CIFAR100
from data.mnist import MNIST
from data.dataloader_clothing1M_one_new import clothing_dataloader
# from model import CNN
import resnet
import argparse, sys
import numpy as np
import datetime
import shutil
import umap

from loss import *

parser = argparse.ArgumentParser()
# parser.add_argument('--lr', type = float, default = 0.001)
parser.add_argument('--result_dir', type=str, help='dir to save result txt files', default='results/')
parser.add_argument('--noise_rate', type=float, help='corruption rate, should be less than 1', default=0.5)
parser.add_argument('--forget_rate', type=float, help='forget rate', default=None)
parser.add_argument('--noise_type', type=str, help='[pairflip, symmetric]', default='symmetric')
parser.add_argument('--num_gradual', type=int, default=10,
                    help='how many epochs for linear drop rate, can be 5, 10, 15. This parameter is equal to Tk for R(T) in Co-teaching paper.')
parser.add_argument('--exponent', type=float, default=1,
                    help='exponent of the forget rate, can be 0.5, 1, 2. This parameter is equal to c in Tc for R(T) in Co-teaching paper.')
parser.add_argument('--top_bn', action='store_true')
parser.add_argument('--dataset', type=str, help='mnist, cifar10, or cifar100', default='clothing1M')
parser.add_argument('--n_epoch', type=int, default=100)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--print_freq', type=int, default=50)
parser.add_argument('--num_workers', type=int, default=4, help='how many subprocesses to use for data loading')
parser.add_argument('--num_iter_per_epoch', type=int, default=2000)
parser.add_argument('--epoch_decay_start', type=int, default=80)

parser.add_argument('--alpha', default=0.8, type=float)
parser.add_argument('--beta', default=0.19, type=float)
parser.add_argument('--gamma', default=0.01, type=float)

parser.add_argument('--lr', default=0.0008, type=float, help='learning_rate')
parser.add_argument('--momentum', default=0.2, type=float, help='momentum')
parser.add_argument('--batch_size', default=128, type=int, help='batch_size')
parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight_decay')
parser.add_argument('--start', type=int, default=64)
parser.add_argument('--step', type=int, default=1)
parser.add_argument('--w', type=float, default=0.9)
parser.add_argument('--num_time', default=1, type=int)
parser.add_argument('--es', default=30, type=int)

# parser.add_argument('--samp_num', default=1037497, type=int)
parser.add_argument('--samp_num', default=256000, type=int)
parser.add_argument('--p_threshold', default=0.95, type=float, help='clean probability threshold')
parser.add_argument('--d_threshold', default=0.95, type=float, help='clean probability threshold')
args = parser.parse_args()

# Seed
# torch.manual_seed(args.seed)
# torch.cuda.manual_seed(args.seed)

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

    train_dataset = SVHN(root='./data/SVHN/',
                         download=False,
                         train=True,
                         transform=transforms.ToTensor(),
                         noise_type=args.noise_type,
                         noise_rate=args.noise_rate
                         )

    test_dataset = SVHN(root='./data/SVHN/',
                        download=False,
                        train=False,
                        transform=transforms.ToTensor(),
                        noise_type=args.noise_type,
                        noise_rate=args.noise_rate
                        )

if args.dataset == 'F-MNIST':
    input_channel = 1
    num_classes = 10
    args.top_bn = False
    args.epoch_decay_start = 80
    args.n_epoch = 200

    train_dataset = Fashion(root='./data/F-MNIST/',
                            download=True,
                            train=True,
                            transform=transforms.ToTensor(),
                            noise_type=args.noise_type,
                            noise_rate=args.noise_rate
                            )

    test_dataset = Fashion(root='./data/F-MNIST/',
                           download=True,
                           train=False,
                           transform=transforms.ToTensor(),
                           noise_type=args.noise_type,
                           noise_rate=args.noise_rate
                           )
# load dataset

if args.dataset == 'mnist':
    input_channel = 1
    num_classes = 10
    args.top_bn = False
    args.epoch_decay_start = 80
    # args.n_epoch = 200
    train_dataset = MNIST(root='./data/',
                          download=True,
                          train=True,
                          transform=transforms.ToTensor(),
                          noise_type=args.noise_type,
                          noise_rate=args.noise_rate
                          )

    test_dataset = MNIST(root='./data/',
                         download=True,
                         train=False,
                         transform=transforms.ToTensor(),
                         noise_type=args.noise_type,
                         noise_rate=args.noise_rate
                         )

if args.dataset == 'cifar10':
    input_channel = 3
    num_classes = 10
    args.top_bn = False
    args.epoch_decay_start = 80
    # args.n_epoch = 200
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

if args.dataset == 'cifar100':
    input_channel = 3
    num_classes = 100
    args.top_bn = False
    args.epoch_decay_start = 100
    # args.n_epoch = 200
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
if args.dataset == 'clothing1M':
    input_channel = 3
    num_classes = 14
    args.top_bn = False
    args.epoch_decay_start = 2
    loader = clothing_dataloader(root="/home/lijiawei/datasets/clothing1M",
                                 batch_size=args.batch_size, num_workers=12,num_batches=args.num_iter_per_epoch)
    # train_dataset, test_dataset = loader.getDataSet()


        # Data Loader (Input Pipeline)
    print('loading dataset...')
    train_loader, noisy_label = loader.run('train')
    # train_eval_loader,_ = loader.run('eval_train')
    test_loader = loader.run('test')
    # val_loader = loader.run('val')
    dataSize = len(noisy_label)
    # train_loader = torch.utils.data.DataLoader(dataset=train_dataset,  # <class 'data.cifar.CIFAR10'>
    #                                            batch_size=args.batch_size,
    #                                            num_workers=args.num_workers,
    #                                            # how many subprocesses to use for data loading 几个并行进程？
    #                                            drop_last=True,
    #                                            shuffle=True)  # 是否打乱
    # test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
    #                                           batch_size=args.batch_size,
    #                                           num_workers=args.num_workers,
    #                                           drop_last=True,
    #                                           shuffle=False)
if args.forget_rate is None:
    forget_rate = args.noise_rate
else:
    forget_rate = args.forget_rate

# noise_or_not = train_dataset.noise_or_not

# clean_label = np.asarray(train_dataset.train_labels.reshape(-1))

# Adjust learning rate and betas for Adam Optimizer
mom1 = 0.9
mom2 = 0.1
alpha_plan = [learning_rate] * args.n_epoch
beta1_plan = [mom1] * args.n_epoch
for i in range(args.epoch_decay_start, args.n_epoch):
    alpha_plan[i] = float(args.n_epoch - i) / (args.n_epoch - args.epoch_decay_start) * learning_rate
    beta1_plan[i] = mom2

save_dir = '/home/wangxiaoxiao/FPL/results/FSE4' + '/'
if not os.path.exists(save_dir):
    os.system('mkdir -p %s' % save_dir)
model_str = args.dataset + '_' + args.noise_type + '_' + str(args.noise_rate) + '_' + str(args.num_time)
txtfile = save_dir + "/" + model_str + ".txt"



# Data Loader (Input Pipeline)
# print('loading dataset...')
# train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
#                                                batch_size=batch_size,
#                                                num_workers=args.num_workers,
#                                                drop_last=True,
#                                                shuffle=True)
#
# test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
#                                               batch_size=batch_size,
#                                               num_workers=args.num_workers,
#                                               drop_last=True,
#                                               shuffle=False)

# noisy_label = np.asarray(train_dataset.train_labels)

# noisy_label = np.asarray(train_dataset.train_labels).tolist().values()
# noisy_label=np.array(list(noisy_label))

# noisy_label = np.asarray(train_dataset.train_noisy_labels)
# noisy_label=noisy_label.values()
# print(noisy_label)
# noisy_label=noisy_label[0]

def adjust_learning_rate(optimizer, epoch):
    # for param_group in optimizer.param_groups:
    #     param_group['lr'] = alpha_plan[epoch]
    #     param_group['betas'] = (beta1_plan[epoch], 0.999)  # Only change beta1
    for param_group in optimizer.param_groups:
        if epoch <= 5:
            param_group['lr'] = 0.0008
            param_group['betas'] = (beta1_plan[epoch], 0.999)
        elif epoch <= 10:
            param_group['lr'] = 0.0005
            param_group['betas'] = (beta1_plan[epoch], 0.999)
        else:
            param_group['lr'] = 0.00005
            param_group['betas'] = (beta1_plan[epoch], 0.999)


# define drop rate schedule
rate_schedule = np.ones(args.n_epoch) * forget_rate
rate_schedule[:args.num_gradual] = np.linspace(0, forget_rate ** args.exponent, args.num_gradual)

def accuracy(logit, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    output = F.softmax(logit, dim=1)
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


# Train the Model
def train(train_loader, soft_labels, epoch, model1, optimizer1,feature):
    # print ('Training %s...' % model_str)
    pure_ratio_list = []
    pure_ratio_1_list = []

    train_total = 0
    train_correct = 0

    feature_labels = torch.zeros(noisy_label.shape[0], 14, dtype=torch.float).cuda(non_blocking=True)


    PL = []

    if epoch < args.es:
        print("WARM UP ING")
    else:
        print("MAKING SOFT LABEL")

        num_samples = args.samp_num
        all_targets = torch.zeros(args.samp_num)
        normalized_features = torch.zeros((num_samples, 512))
        predictions = torch.zeros(num_samples)
        with torch.no_grad():
            for i, (images, labels, indexes) in enumerate(train_loader):
                ind = indexes.cpu().numpy().transpose()
                if i > args.num_iter_per_epoch:
                    break
                images = Variable(images).cuda()
                labels = Variable(labels).cuda()
                predict, feature = model1(images)
                # l_c1=nn.Softplus()
                # feature=l_c1(feature)
                normalized_fea=F.normalize(feature, dim=1)

                _, predicted = torch.max(predict, 1)

                for b in range(images.size(0)):
                    # all_targets[indexes[b]] = labels[b]
                    all_targets[indexes[b]] = np.argmax(soft_labels[indexes[b]].cpu(), axis=-1).cuda()
                    normalized_features[indexes[b]] = normalized_fea[b]
                    predictions[indexes[b]] = predicted[b]
        #
        overall_distance = torch.zeros((num_samples,))
        all_prob = np.zeros((num_samples,))
        centers = torch.zeros((14, 512))
        for cls_ in range(14):
            centers[cls_] = F.normalize(normalized_features[predictions == cls_].mean(dim=0), dim=0)

        for cls_ in range(14):
            distance_cls = (normalized_features[all_targets == cls_] * centers[cls_]).sum(dim=1)
            overall_distance[all_targets == cls_] = distance_cls

        overall_distance = (overall_distance - overall_distance.min()) / (
                    overall_distance.max() - overall_distance.min())
        overall_distance = overall_distance.numpy().reshape(-1, 1)

        # gmm = GaussianMixture(n_components=2, max_iter=100, tol=1e-1, reg_covar=5e-4)
        # gmm.fit(overall_distance)
        #
        # gmm_prob = gmm.predict_proba(overall_distance)
        # gmm_prob = gmm_prob[:, gmm.means_.argmax()]

        center_t=centers.permute(1,0)

        overall_distance= torch.mm(normalized_features, center_t)

        f_label=np.argmax(overall_distance.cpu(), axis=-1)
        feature_labels[torch.arange(noisy_label.shape[0]), f_label] = 1


    for i, (images, labels, indexes) in enumerate(train_loader):
        ind = indexes.cpu().numpy().transpose()
        if i > args.num_iter_per_epoch:
            break
        images = Variable(images).cuda()
        labels = Variable(labels).cuda()
        # print("epoch：", epoch, "的第", i, "个inputs", images.data.size(), "labels", labels.data)
        # Forward + Backward + Optimize
        logits1,_= model1(images)
        prec1, _ = accuracy(logits1, labels, topk=(1, 5))
        train_total += 1
        train_correct += prec1

        if epoch < args.es:
            loss_1 = F.cross_entropy(logits1, labels)
            optimizer1.zero_grad()
            loss_1.backward()
            optimizer1.step()
        else:
            # clean=[]
            # for i in range(0,1000):
            #     if gmm_prob[indexes[i]]>args.p_threshold:
            #         clean.append(i)

            # print(clean)
            prob_1 = F.softmax(logits1.detach(), dim=1)
            # prob_1.reshape(-1,14)
            # print(soft_labels[ind].shape)
            # print(prob_1.shape)
            # print(feature_labels[ind].shape)
            soft_labels[ind] = args.alpha * soft_labels[ind] + args.beta * prob_1 + args.gamma * feature_labels[ind]
            # obtain weights
            weights, _ = soft_labels[indexes].max(dim=1)
            weights *= logits1.shape[0] / weights.sum()

            # compute cross entropy loss, without reduction
            loss = torch.sum(-F.log_softmax(logits1, dim=1) * soft_labels[indexes], dim=1)
            # loss = torch.sum(-F.log_softmax(logits1[clean], dim=1) * soft_labels[indexes[clean]], dim=1)

            # sample weighted mean
            loss = (loss * weights).mean()
            # loss_1 = torch.sum(-F.log_softmax(logits1, dim=1) * soft_labels[indexes], dim=1)
            # loss_1 = torch.sum(-F.log_softmax(logits1[clean], dim=1) * soft_labels[indexes[clean]], dim=1)
            # loss_1 = torch.sum(-F.log_softmax(logits1, dim=1) * labels[indexes[clean]], dim=1)
            # loss_1 = torch.mean(loss)

            optimizer1.zero_grad()
            loss.backward()
            optimizer1.step()



            # loss_1 = F.cross_entropy(logits1[clean], labels[clean])
            # prob_1 = F.softmax(logits1.detach(), dim=1)
            # soft_labels[ind] = 0.8 * soft_labels[ind] + 0.1 * prob_1 + 0.1 * feature_labels[ind]
            # loss_1 = torch.sum(-F.log_softmax(logits1, dim=1) * soft_labels[indexes], dim=1)
            # loss_1 = torch.mean(loss_1)
            # optimizer1.zero_grad()
            # loss_1.backward()
            # optimizer1.step()



    train_acc1 = float(train_correct) / float(train_total)

    return train_acc1, soft_labels,feature


# Evaluate the Model
def evaluate(test_loader, model1):
    # print ('Evaluating %s...' % model_str)
    model1.eval()  # Change model to 'eval' mode.
    with torch.no_grad():
        correct1 = 0
        total1 = 0
        for images, labels in test_loader:
            images = Variable(images).cuda()
            logits1,_= model1(images)
            outputs1 = F.softmax(logits1, dim=1)
            _, pred1 = torch.max(outputs1.data, 1)
            total1 += labels.size(0)
            correct1 += (pred1.cpu() == labels).sum()
    acc1 = 100 * float(correct1) / float(total1)
    return acc1


def main():


    #Feature
    feature = torch.zeros(noisy_label.shape[0], 128, dtype=torch.float).cuda(non_blocking=True)

     # soft_labels
    if args.dataset == 'cifar100':
        soft_labels = torch.zeros(noisy_label.shape[0], 100, dtype=torch.float).cuda(non_blocking=True)
        soft_labels[torch.arange(noisy_label.shape[0]), noisy_label] = 1
    else:
        soft_labels = torch.zeros(noisy_label.shape[0], 14, dtype=torch.float).cuda(non_blocking=True)
        soft_labels[torch.arange(noisy_label.shape[0]), noisy_label] = 1



    print('building model...')
    cnn1 = resnet.resnet18(pretrained=True)
    cnn1.fc = nn.Linear(512, 14)
    # cnn1 = CNN(input_channel=input_channel, n_outputs=num_classes)
    # cnn1.load_state_dict(torch.load(save_root1))

    cnn1.cuda()
    # print cnn1.parameters
    optimizer1 = torch.optim.Adam(cnn1.parameters(), lr=learning_rate)
    # criterion = SelfAdaptiveTrainingCE(labels=targets, num_classes=10)

    mean_pure_ratio1 = 0

    save_dir = '/home/wangxiaoxiao/FPL/results/FSE4' + '/'
    if not os.path.exists(save_dir):
        os.system('mkdir -p %s' % save_dir)
    model_str = args.dataset + '_' + args.noise_type + '_' + str(args.noise_rate) + '_' + str(args.num_time)
    txtfile = save_dir + "/" + model_str + ".txt"

    epoch = 0
    train_acc1 = 0

    best_test_acc1 = 0

    # evaluate models with random weights
    test_acc1 = evaluate(test_loader, cnn1)
    print('Epoch [%d/%d] Test Accuracy: Model1 %.4f %%  Pure Ratio1 %.4f %% ' % (
        epoch + 1, args.n_epoch, test_acc1, mean_pure_ratio1))

    with open(txtfile, "a") as myfile:
        myfile.write(str(int(epoch)) + ' ' + str(train_acc1) + ' ' + str(test_acc1) + ' ' + str(mean_pure_ratio1)  + "\n")

    #  save results
    # with open(txtfile, "a") as myfile:
    #     myfile.write(str(int(epoch)) + ': '  + str(train_acc1) +' '  + str(train_acc2) +' '  + str(test_acc1) + " " + str(test_acc2) + ' '  + str(mean_pure_ratio1) + ' '  + str(mean_pure_ratio2) + "\n")

    # training
    Last10_1 = []
    Last10_2 = []
    for epoch in range(1, args.n_epoch):
        # train models
        cnn1.train()
        adjust_learning_rate(optimizer1, epoch)

        train_acc1, soft_labels, feature = train(train_loader, soft_labels,epoch, cnn1, optimizer1,feature)
        # evaluate models
        test_acc1 = evaluate(test_loader, cnn1)


        if test_acc1 > best_test_acc1:
            best_test_acc1 = test_acc1

        # save results
        # mean_pure_ratio1 = sum(pure_ratio_1_list) / len(pure_ratio_1_list)
        # mean_pure_ratio2 = sum(pure_ratio_2_list) / len(pure_ratio_2_list)

        # print('Epoch [%d/%d] Test Accuracy: Model1 %.4f %% Model2 %.4f %% Pure Ratio1 %.4f %% Pure Ratio2 %.4f %%' % (
        #     epoch + 1, args.n_epoch, test_acc1, test_acc2, mean_pure_ratio1, mean_pure_ratio2))

        print('Epoch [%d/%d] Test Acc: Model1 %.4f %% , Best Test Acc: Model1 %.4f %%' % (
        epoch + 1, args.n_epoch, test_acc1, best_test_acc1))

        with open(txtfile, "a") as myfile:
            myfile.write(
                str(int(epoch)) + ' ' + str(train_acc1) + ' ' + str(test_acc1) + "\n")

        if epoch >= args.n_epoch - 10:
            Last10_1.append(test_acc1)


        if epoch == args.n_epoch - 1:
            print('Last10_1:')
            print(Last10_1)
            print('mean±std: %.2f±%.2f' % (np.mean(Last10_1), np.std(Last10_1)))


#        with open(txtfile, "a") as myfile:
#            myfile.write(str(int(epoch)) + ': '  + str(train_acc1) +' '  + str(train_acc2) +' '  + str(test_acc1) + " " + str(test_acc2) + ' ' + str(mean_pure_ratio1) + ' ' + str(mean_pure_ratio2) + "\n")


if __name__ == '__main__':
    main()
