from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image

class clothing_dataset(Dataset): 
    def __init__(self, transform, mode): 
        self.train_imgs = []
        self.test_imgs = []
        self.val_imgs = []
        self.train_labels = {}
        self.test_labels = {}
        self.val_labels = {}
        self.transform = transform
        self.mode = mode
        with open('/home/lijiawei/datasets/clothing/noisy_train_key_list.txt','r') as f:
            lines = f.read().splitlines()
        for l in lines:
            img_path = '/home/lijiawei/datasets/clothing/'+l[7:]
            self.train_imgs.append(img_path)

        with open('/home/lijiawei/datasets/clothing/clean_test_key_list.txt','r') as f:
            lines = f.read().splitlines()
        for l in lines:
            img_path = '/home/lijiawei/datasets/clothing/'+l[7:]
            self.test_imgs.append(img_path)

        with open('/home/lijiawei/datasets/clothing/clean_val_key_list.txt','r') as f:
            lines = f.read().splitlines()
        for l in lines:
            img_path = '/home/lijiawei/datasets/clothing/'+l[7:]
            self.val_imgs.append(img_path)
            
        with open('/home/lijiawei/datasets/clothing/noisy_label_kv.txt','r') as f:
            lines = f.read().splitlines()
        for l in lines:
            entry = l.split()           
            img_path = '/home/lijiawei/datasets/clothing/'+entry[0][7:]
            self.train_labels[img_path] = int(entry[1])

        with open('/home/lijiawei/datasets/clothing/clean_label_kv.txt','r') as f:
            lines = f.read().splitlines()
        for l in lines:
            entry = l.split()           
            img_path = '/home/lijiawei/datasets/clothing/'+entry[0][7:]
            self.test_labels[img_path] = int(entry[1])


            
    def __getitem__(self, index):  
        if self.mode=='train':
            img_path = self.train_imgs[index]
            target = self.train_labels[img_path]

        elif self.mode=='test':
            img_path = self.test_imgs[index]
            target = self.test_labels[img_path]
        elif self.mode=='val':
            img_path = self.val_imgs[index]
            target = self.test_labels[img_path]            
        image = Image.open(img_path).convert('RGB')    
        img = self.transform(image)
        return img, target,index
    
    def __len__(self):
        if self.mode=='train':
            return len(self.train_imgs)
        elif self.mode=='test':
            return len(self.test_imgs)      
        elif self.mode=='val':
            return len(self.val_imgs)           
        
class clothing_dataloader():  
    def __init__(self, batch_size, num_workers, shuffle):
    
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
   

        self.transform_train = transforms.Compose([
                transforms.Resize(256),
                #transforms.RandomSizedCrop(224),
                #transforms.RandomHorizontalFlip(),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
            ]) # meanstd transformation

        self.transform_test = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
            ])    
        self.train_dataset = clothing_dataset(transform=self.transform_train, mode='train')
        self.test_dataset = clothing_dataset(transform=self.transform_test, mode='test')
        self.val_dataset = clothing_dataset(transform=self.transform_test, mode='val')


    def getDataSet(self):

        return self.train_dataset, self.val_dataset, self.test_dataset
