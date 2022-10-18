# GCL: Long-tailed Visual Recognition via Gaussian Clouded Logit Adjustment
This is the source code for our CVPR (2022) paper: [Long-tailed Visual Recognition via Gaussian Clouded Logit Adjustment](https://openaccess.thecvf.com/content/CVPR2022/html/Li_Long-Tailed_Visual_Recognition_via_Gaussian_Clouded_Logit_Adjustment_CVPR_2022_paper.html) based on Pytorch. 
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

## Results and Models for Large-scale Datasets ()

* Stage-1:

| Dataset     | Arch       | Top-1 Accuracy | Log           | Model |
| ----------- | ---------- | -------------- | ------------- | ----- |
| ImageNet-LT | ResNet-50  | 45.5%          | 7.98%         | [link](https://drive.google.com/file/d/1QKVnK7n75q465ppf7wkK4jzZvZJE_BPi/view?usp=sharing)  |
| iNat 2018   | ResNet-50  | 66.9%          | 5.37%         | [link](https://drive.google.com/file/d/1wvj-cITz8Ps1TksLHi_KoGsq9CecXcVt/view?usp=sharing)  |
| Places-LT   | ResNet-152 | 29.4%          | 16.7%         | [link](https://drive.google.com/file/d/1Tx-tY5Y8_-XuGn9ZdSxtAm0onOsKWhUH/view?usp=sharing)  |

* Stage-2:

| Dataset     | Arch       | Top-1 Accuracy | Log           | Model |
| ----------- | ---------- | -------------- | ------------- | ----- |
| ImageNet-LT | ResNet-50  | 52.7%          | 1.78%         | [link](https://drive.google.com/file/d/1ofJKlUJZQjjkoFU9MLI08UP2uBvywRgF/view?usp=sharing)  |
| iNat 2018   | ResNet-50  | 71.6%          | 7.67%         | [link](https://drive.google.com/file/d/1crOo3INxqkz8ZzKZt9pH4aYb3-ep4lo-/view?usp=sharing)  |
| Places-LT   | ResNet-152 | 40.4%          | 3.41%         | [link](https://drive.google.com/file/d/1DgL0aN3UadI3UoHU6TO7M6UD69QgvnbT/view?usp=sharing)  |

## <a name="Citation"></a>Citation

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
