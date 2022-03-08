from dataset.baseset import BaseSet
import random, cv2
import numpy as np


class iNaturalist(BaseSet):
    def __init__(self, mode='train', cfg=None, transform=None):
        super(iNaturalist, self).__init__(mode, cfg, transform)
        np.random.seed(0)
        random.seed(0)
        self.class_dict = self._get_class_dict()
        
        
    def __getitem__(self, index):
        #if self.mode == 'train': #self.cfg.sampler_type == "weighted sampler" and self.mode == 'train':
            #assert self.cfg.sampler_type in ['balance', 'square', 'progressive']
            #if  self.cfg.sampler_type == "balance":
            #sample_class = random.randint(0, self.num_classes - 1)
            #elif self.cfg.sampler_type == "square":
            #    sample_class = np.random.choice(np.arange(self.num_classes), p=self.square_p)
            #else:
            #sample_class = np.random.choice(np.arange(self.num_classes), p=self.progress_p)
            #sample_indexes = self.class_dict[sample_class]
            #index = random.choice(sample_indexes)
        now_info = self.data[index]
        img = self._get_image(now_info)
        image = self.transform(img)
        #meta = dict()
        image_label = now_info['category_id']  # 0-index
        return image, image_label #, meta

    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(len(self.class_dict)):
            cls_num_list.append(len(self.class_dict[i]))
        return cls_num_list

if __name__ == '__main__':

    import argparse
    import torchvision.transforms as transforms
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--TRAIN_JSON', default='D:\dataset\ImageNet_LT\ImageNet_LT_train.json', type=str, 
                    help='input image size')
    parser.add_argument('--VALID_JSON', default='D:\dataset\ImageNet_LT\ImageNet_LT_val.json', type=str, 
                    help='input image size')
    parser.add_argument('--INPUT_SIZE', default=(224, 224), type=list,
                    help='input image size')
    parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
    cfg = parser.parse_args()

    transform = transforms.Compose([
        transforms.RandomResizedCrop( size=(224, 224),scale=(0.08, 1.0),ratio=(0.75, 1.333333333)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])     

    trainset = iNaturalist(mode='train', cfg=cfg, transform=transform)
        
    trainloader = iter(trainset)
    cls_num_list = trainset.get_cls_num_list()
    data, label = next(trainloader)
    
    print(data.shape)
    print(label.shape)
    








