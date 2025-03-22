# GCL: Long-tailed Visual Recognition via Gaussian Clouded Logit Adjustment 
This is the source code for our CVPR (2022) paper: [Long-tailed Visual Recognition via Gaussian Clouded Logit Adjustment](https://openaccess.thecvf.com/content/CVPR2022/html/Li_Long-Tailed_Visual_Recognition_via_Gaussian_Clouded_Logit_Adjustment_CVPR_2022_paper.html) based on Pytorch. 
This version is a demo of how to use GCL loss. The version that supports more datasets is in the works and is coming soon.

This repository also include the source code for our TAI paper: [Adjusting Logit in Gaussian Form for Long-Tailed Visual Recognition](https://arxiv.org/abs/2305.10648).
The angular form GCL can be found in [losses.py](https://github.com/Keke921/GCLLoss/blob/main/losses.py) GCLAngLoss. 


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

Sorry. I have run out of the storage space and the model parameters are temporarily unavailable. Looking for other disk now.

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


## You May Find Our Additional Works of Interest

* [TPAMI'23] Key Point Sensitive Loss for Long-Tailed Visual Recognition [[paper](https://drive.google.com/file/d/1gOJDHBJ_M7RmU6Iw2p6uXIyo8pNgVMrv/view?pli=1)] [[code](https://github.com/Keke921/KPSLoss)]

* [CVPR'23] Long-Tailed Visual Recognition via Self-Heterogeneous Integration with Knowledge Excavation [[paper](https://arxiv.org/pdf/2304.01279)] [[code](https://github.com/jinyan-06/SHIKE)]

* [AAAI'24] Feature Fusion from Head to Tail for Long-Tailed Visual Recognition [[paper](https://arxiv.org/pdf/2306.06963)] [[code](https://github.com/Keke921/H2T)]

* [NeurIPS'24] Improving Visual Prompt Tuning by Gaussian Neighborhood Minimization for Long-Tailed Visual Recognition [[paper](https://arxiv.org/pdf/2410.21042)] [[code](https://github.com/Keke921/GNM-PT)]


## Other Resources of long-tailed visual recognition
[Awesome-LongTailed-Learning](https://github.com/Vanint/Awesome-LongTailed-Learning)

[Awesome-of-Long-Tailed-Recognition](https://github.com/zwzhang121/Awesome-of-Long-Tailed-Recognition)

[Long-Tailed-Classification-Leaderboard](https://github.com/yanyanSann/Long-Tailed-Classification-Leaderboard)

[BagofTricks-LT](https://github.com/zhangyongshun/BagofTricks-LT)

## Connection
If you have any questions, please send the email to Mengke Li at: csmkli@comp.hkbu.edu.hk.

## Citation
```
@inproceedings{Li2022GCL,
  author    = {Mengke Li, Yiu{-}ming Cheung, Yang Lu},
  title     = {Long-tailed Visual Recognition via Gaussian Clouded Logit Adjustment},
  pages = {6929-6938},
  booktitle = {CVPR},
  year      = {2022},
}
```

```
@article{Li2024AGCL,
  author={Li, Mengke and Cheung, Yiu-ming and Lu, Yang and Hu, Zhikai and Lan, Weichao and Huang, Hui},
  journal={IEEE Transactions on Artificial Intelligence}, 
  title={Adjusting Logit in Gaussian Form for Long-Tailed Visual Recognition}, 
  year={2024},
  volume={5},
  number={10},
  pages={5026-5039},
  doi={10.1109/TAI.2024.3401102}}
```


## Acknowledgment
We refer to some codes from [MisLAS](https://github.com/dvlab-research/MiSLAS). Many thanks to the authors.
