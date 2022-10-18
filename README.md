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

## Results and Models for Large-scale Datasets (temporarily)

(*Note: The code in this repository may need to be modified to match the .pth file. Since I left my previous workplace, there is no GPU retraining model now. I temporarily share previously stored .pth file for the model. Sorry for the trouble.)

* Stage-1:

| Dataset     | Arch       | Top-1 Accuracy | Log           | Model |
| ----------- | ---------- | -------------- | ------------- | ----- |
| ImageNet-LT | ResNet-50  | 52.928%        | [link](https://lifehkbueduhk-my.sharepoint.com/:u:/g/personal/18482244_life_hkbu_edu_hk/EYro8K-qsKJOvkPf3RJrn6oBnd98VXIQlkrCnQLoex-U8Q?e=2TJOaF)        | [link](https://lifehkbueduhk-my.sharepoint.com/:u:/g/personal/18482244_life_hkbu_edu_hk/EYro8K-qsKJOvkPf3RJrn6oBnd98VXIQlkrCnQLoex-U8Q?e=2TJOaF)  |
| iNat 2018   | ResNet-50  | 70.327%        | [link](https://lifehkbueduhk-my.sharepoint.com/:u:/r/personal/18482244_life_hkbu_edu_hk/Documents/Migration%20files/long-tailed-work/LT-Classification-2/shared%20files/iNat%202018/iNat_NoiScr_None_mixup_70.327/iNat-backbone-bs256.log?csf=1&web=1&e=1SDUuz)   | [link]([https://drive.google.com/file/d/1wvj-cITz8Ps1TksLHi_KoGsq9CecXcVt/view?usp=sharing](https://lifehkbueduhk-my.sharepoint.com/:u:/r/personal/18482244_life_hkbu_edu_hk/Documents/Migration%20files/long-tailed-work/LT-Classification-2/shared%20files/iNat%202018/iNat_NoiScr_None_mixup_70.327/iNat-backbone-bs256.log?csf=1&web=1&e=1SDUuz))  |
| Places-LT   | ResNet-152 | 29.4%          | 16.7%         | [link](https://drive.google.com/file/d/1Tx-tY5Y8_-XuGn9ZdSxtAm0onOsKWhUH/view?usp=sharing)  |

* Stage-2:

| Dataset     | Arch       | Top-1 Accuracy | Log           | Model |
| ----------- | ---------- | -------------- | ------------- | ----- |
| ImageNet-LT | ResNet-50  | 54.884%        | [link (train)](https://lifehkbueduhk-my.sharepoint.com/:x:/g/personal/18482244_life_hkbu_edu_hk/EdjYUsWSEyhHih_77ETKo6QBffmR0_weBek8sXuT2E6SBQ?e=IHQ2mz)  [link (val)](https://lifehkbueduhk-my.sharepoint.com/:x:/g/personal/18482244_life_hkbu_edu_hk/EXzcoAhffupAjgq2UidEBSMBxuT5g8C2GmFjSsvQ2gpmpg)| [link](https://lifehkbueduhk-my.sharepoint.com/:u:/g/personal/18482244_life_hkbu_edu_hk/EfS6Y3e0AvlCg4Gawwcoo7QBpHPrN4ckDylxaAfIvHoJiA)  |
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
