# Feature-Balanced Loss for Long-Tailed Visual Recognition
This is the official source code for the Paper: **Feature-Balanced Loss for Long-Tailed Visual Recognition** based on Pytorch. Recommending use `>exp.log 2>&1` or `2>&1 | tee` to record historical standard output. 

## CIFAR10
```bash
$ python train.py --arch resnet32 /
                  --dataset cifar10 --data_path './dataset/data_img' /
                  --gpu 3 /
                  --loss_type 'FeaBal' --batch_size 64 --learning_rate 0.1 --lambda_ 60
```
## CIFAR100
```bash
$ python train.py --arch resnet32 /
                  --dataset cifar100 --data_path './dataset/data_img' /
                  --gpu 3 /
                  --loss_type 'FeaBal' --batch_size 64 --learning_rate 0.1 --lambda_ 60
```
## ImageNet
```bash
$ python train.py --arch resnet50 / 
                  --dataset imagenet --data_path './dataset/data_txt' --img_path '/home/datasets/imagenet/ILSVRC2012_dataset' / 
                  --gpu 3 /
                  --loss_type 'FeaBal' --batch_size 512 --learning_rate 0.2 --lambda_ 150
```
## iNaturalist
```bash
$ python train.py --arch resnet50 / 
                  --dataset inat --data_path './dataset/data_txt' --img_path '/home/datasets/iNaturelist2018' / 
                  --gpu 3 /
                  --loss_type 'FeaBal' --batch_size 512  --learning_rate 0.2 --lambda_ 150
```

## Places
```bash
$ python train.py --arch resnet152_p / 
                  --dataset place365 --data_path './dataset/data_txt' --img_path '/home/datasets/Places365' /
                  --gpu 3 /
                  --loss_type 'FeaBal' --batch_size 512  --learning_rate 0.2 --lambda_ 150
```