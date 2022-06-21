# GCL: Long-tailed Visual Recognition via Gaussian Clouded Logit Adjustment
This is the source code for our CVPR (2022) paper: [Long-tailed Visual Recognition via Gaussian Clouded Logit Adjustment](https://www.techrxiv.org/articles/preprint/Long-tailed_Visual_Recognition_via_Gaussian_Clouded_Logit_Adjustment/17031920) based on Pytorch. 
This version is a demo of how to use GCL loss. The version that supports more datasets is in the works and is coming soon.

## CIFAR10
First stage
```bash
$ python cifar_train_backbone.py --arch resnet32 /
                                 --dataset cifar10 --data_path './dataset/data_img' /
                                 --loss_type 'GCL' --imb_factor 0.01 /
                                 --batch_size 64 --learning_rate 0.1 
```
Second stage
```bash
$ python cifar_train_classifier.py --arch resnet32 /
                                 --dataset cifar10 --data_path './dataset/data_img' /
                                 --loss_type 'GCL' --imb_factor 0.01 /
                                 --train_rule 'BalancedRS'/
                                 --batch_size 64 --learning_rate 0.1 
```


## To do list:
- [x] Support Cifar10/100-LT dataset
- [ ] Support imageNet-LT
- [ ] Support iNaturalist2018
- [ ] Support Places365-LT
- [ ] More loss functions
- [ ] Separate configuration files for easier execution
- [ ] Some other minor performance improvements


## Citation
```
@inproceedings{Li2022Long,
  author    = {Mengke Li, Yiu{-}ming Cheung, Yang Lu},
  title     = {Long-tailed Visual Recognition via Gaussian Clouded Logit Adjustment},
  pages = {6929-6938},
  booktitle = {CVPR},
  year      = {2022},
}
```

## Other Resources of long-tailed visual recognition
[Awesome-LongTailed-Learning](https://github.com/Vanint/Awesome-LongTailed-Learning)

[Awesome-of-Long-Tailed-Recognition](https://github.com/zwzhang121/Awesome-of-Long-Tailed-Recognition)

[Long-Tailed-Classification-Leaderboard](https://github.com/yanyanSann/Long-Tailed-Classification-Leaderboard)

[BagofTricks-LT](https://github.com/zhangyongshun/BagofTricks-LT)

## Connection
If you have any questions, please send the email to Mengke Li at: csmkli@comp.hkbu.edu.hk.
