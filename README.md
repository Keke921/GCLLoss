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

*Note: I have modified the code several times. The code in this repository may need to be modified (Mainly the backbone and classifier loading parts, the classifier network layer names may be inconsistent.) to match the pth file. Since I left my previous workplace, there is no GPU retraining model now. I temporarily share previously stored pth file for the model. Sorry for the inconvience and hope you can understand.

* Stage-1:

| Dataset     | Arch       | Top-1 Accuracy | Log           | Model |
| ----------- | ---------- | -------------- | ------------- | ----- |
| ImageNet-LT | ResNet-50  | 52.928%        | [link](https://lifehkbueduhk-my.sharepoint.com/:u:/g/personal/18482244_life_hkbu_edu_hk/EYro8K-qsKJOvkPf3RJrn6oBnd98VXIQlkrCnQLoex-U8Q?e=2TJOaF)        | [link](https://lifehkbueduhk-my.sharepoint.com/:u:/g/personal/18482244_life_hkbu_edu_hk/EYro8K-qsKJOvkPf3RJrn6oBnd98VXIQlkrCnQLoex-U8Q?e=2TJOaF)  |
| iNat 2018   | ResNet-50  | 70.327%        | [link](https://lifehkbueduhk-my.sharepoint.com/:u:/g/personal/18482244_life_hkbu_edu_hk/EfVPDmTDauhHvx8ys0-QKHABEJt0hFZtyn_7HYRxekiTUQ?e=Uhat9r)       | [link](https://lifehkbueduhk-my.sharepoint.com/:u:/g/personal/18482244_life_hkbu_edu_hk/EbCsmx-xbg9Aq2m8sRUsMGMBxyxprq1xTmsjlAjqJFd9lQ?e=B9Pojb) |
| Places-LT   | ResNet-152 | 34.589%        | [link](https://lifehkbueduhk-my.sharepoint.com/:u:/g/personal/18482244_life_hkbu_edu_hk/ERdVRvw1a6tFkxXFRsMLSWIB5PVqjzQ_J_Lejct96r1eGQ?e=DOghYk)        | [link](https://lifehkbueduhk-my.sharepoint.com/:u:/g/personal/18482244_life_hkbu_edu_hk/EeZKudpg0WVAm0LDkY2EIzMBIA88fzyUobI4UCY5wkP4tg)  |

* Stage-2:

| Dataset     | Arch       | Top-1 Accuracy | Log           | Model |
| ----------- | ---------- | -------------- | ------------- | ----- |
| ImageNet-LT | ResNet-50  | 54.884%        | [link (train)](https://lifehkbueduhk-my.sharepoint.com/:x:/g/personal/18482244_life_hkbu_edu_hk/EdjYUsWSEyhHih_77ETKo6QBffmR0_weBek8sXuT2E6SBQ?e=IHQ2mz) [link (val)](https://lifehkbueduhk-my.sharepoint.com/:x:/g/personal/18482244_life_hkbu_edu_hk/EXzcoAhffupAjgq2UidEBSMBxuT5g8C2GmFjSsvQ2gpmpg)| [link](https://lifehkbueduhk-my.sharepoint.com/:u:/g/personal/18482244_life_hkbu_edu_hk/EfS6Y3e0AvlCg4Gawwcoo7QBpHPrN4ckDylxaAfIvHoJiA)  |
| iNat 2018   | ResNet-50  | 72.005%        | [link](https://lifehkbueduhk-my.sharepoint.com/:u:/g/personal/18482244_life_hkbu_edu_hk/Edg2j7yW-HRMi4jrrbE0n70BCUZ9_L82pTyek9yp60cwUQ?e=kJxfUJ)    | [link](https://lifehkbueduhk-my.sharepoint.com/:u:/g/personal/18482244_life_hkbu_edu_hk/EbFwr8fJNdxFiPfoE64-j5UB4e0MKxOZgdLQ_qACR9tsbA?e=YvRnv2)   |
| Places-LT   | ResNet-152 | 40.641%          | [link](https://lifehkbueduhk-my.sharepoint.com/:u:/g/personal/18482244_life_hkbu_edu_hk/EXucyzMKExlFlcCDEif8mF4BcFNMt7M50igHF_C7IFwjqg?e=0eiFaa)  | [link](https://lifehkbueduhk-my.sharepoint.com/:u:/g/personal/18482244_life_hkbu_edu_hk/Eexoqm9t4ylGqKNW2N_0LFsBgEz-_NtUdRAYHZyyRPfHWQ?e=2xfKXh)  |

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
