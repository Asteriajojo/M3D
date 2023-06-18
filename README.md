# Minimizing Maximum Model Discrepancy for Transferable Black-box Targeted Attacks (CVPR'23)

This is a pytorch implementation of Minimizing Maximum Model Discrepancy for Transferable Black-box Targeted Attacks.

[Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhao_Minimizing_Maximum_Model_Discrepancy_for_Transferable_Black-Box_Targeted_Attacks_CVPR_2023_paper.pdf) 

> **Abstract:** *In this work, we study the black-box targeted attack problem from the model discrepancy perspective. On the theoretical side, we present a generalization error bound for black-box targeted attacks, which gives a rigorous theoretical analysis for guaranteeing the success of the attack. We reveal that the attack error on a target model mainly depends on empirical attack error on the substitute model and the maximum model discrepancy among substitute models. On the algorithmic side, we derive a new algorithm for black-box targeted attacks based on our theoretical analysis, in which we additionally **m**inimize the **m**aximum **m**odel **d**iscrepancy(M3D) of the substitute models when training the generator to generate adversarial examples. In this way, our model is capable of crafting highly transferable adversarial examples that are robust to the model variation, thus improving the success rate for attacking the black-box model. We conduct extensive experiments on the ImageNet dataset with different classification models, and our proposed approach outperforms existing state-of-the-art methods by a significant margin. *
> 


## Citation
If you find our work, this repository and pretrained adversarial generators useful. Please consider giving a star :star: and cite our work.
```bibtex
@inproceedings{zhao2023minimizing,
  title={Minimizing Maximum Model Discrepancy for Transferable Black-box Targeted Attacks},
  author={Zhao, Anqi and Chu, Tong and Liu, Yahao and Li, Wen and Li, Jingjing and Duan, Lixin},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={8153--8162},
  year={2023}
}
```

### Contents  
1) [Contributions](#Contributions) 
2) [Acknowledge](#Acknowledge)
3) [Pretrained Targeted Generator](#Pretrained-Targeted-Generator) 
4) [Installation](#Installation) 
5) [Training](#Training)
6) [The Implementation Details of Discriminators](#The-Implementation-Details-of-Discriminators)
6) [Evaluation](#Evaluation)


## Contributions

1. We present a generalization error bound for black-box
targeted attacks based on the model discrepancy perspective.
2. We design a novel generative approach called Minimizing Maximum Model Discrepancy (M3D) attack
to craft adversarial examples with high transferability
based on the generalization error bound.
3. We demonstrate the effectiveness of our method by
strong empirical results, where our approach outperforms the state-of-art methods by a significant margin.


## Acknowledge

<sup>([top](#contents))</sup>
Code adapted from [TTP](https://github.com/Muzammal-Naseer/TTP). We thank them for their wonderful code base. 


## Pretrained Targeted Generator
<sup>([top](#contents))</sup> 
*If you find our pretrained Adversarial Generators useful, please consider [citing our work](#Citation).*

Class to Label Mapping

```
Class Number: Class Name
24: Great Grey Owl
99: Goose
245: French Bulldog
344: Hippopotamus
471: Cannon
555: Fire Engine
661: Model T
701: Parachute
802: Snowmobile
919: Street Sign       
```

This is how the pretrianed generators are saved: _"netG_Discriminator_epoch_targetLabel.pth" e.g., netG_vgg19_bn_9_919.pth means that generator is trained agisnt vgg19_bn (Discriminator) for 10 epoch by M3D and the target label is 919(Street Sign).

|Source Model|24|99|245|344|471|555|661|701|802|919|
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|VGG19_BN|[Grey Owl](https://drive.google.com/file/d/1E96nesNUGwGmsb2yIJazSS9ul4f5qGdi/view?usp=sharing)|[Goose](https://drive.google.com/file/d/1T7QS4okMGgfvIrlxLiG92cU8zom3ERhr/view?usp=sharing)|[French Bulldog](https://drive.google.com/file/d/1aCF4CW5RmqQGc3hbLbFStqneGBMqNMBc/view?usp=sharing)|[Hippopotamus](https://drive.google.com/file/d/1dCqImVApJko8A92tL-HTaCbagbqltANu/view?usp=sharing)|[Cannon](https://drive.google.com/file/d/1uBRxGuJ-V5IH1BYtY9aKaHcavbLyLgCx/view?usp=sharing)|[Fire Engine](https://drive.google.com/file/d/1HmmzvF9wBxUNucJ2a4rzt8H1wPGGw4_A/view?usp=sharing)|[Model T](https://drive.google.com/file/d/1tZMisg8zn7--Y4VHHFpNupDUklfUYW4N/view?usp=sharing)|[Parachute](https://drive.google.com/file/d/1r9uIu5KC9Fe56gBxp06IB0URAUmIqDEA/view?usp=sharing)|[Snowmobile](https://drive.google.com/file/d/1MAhkQb51OaHawunNHEi7500lGkONx8FC/view?usp=sharing)|[Street Sign](https://drive.google.com/file/d/1TGdNVCi-PCBl81KiePp6XZBSAbnbiLkw/view?usp=sharing)|
|ResNet50|[Grey Owl](https://drive.google.com/file/d/1JOQvXaGz23KU33bNt7ePKi53pL0WtyaC/view?usp=sharing)|[Goose](https://drive.google.com/file/d/1qNx41n483k7jMlgUdpBWOcmrR-nSfPUX/view?usp=sharing)|[French Bulldog](https://drive.google.com/file/d/17LKZXvyHBPNL3t2dGy462Ygzark-fsK3/view?usp=sharing)|[Hippopotamus](https://drive.google.com/file/d/1caQkHfcU0iSviAFCZjWrbxwNEqSl6oXx/view?usp=sharing)|[Cannon](https://drive.google.com/file/d/1bkSMZi8CB0OqoTMINFG36SrSEgDhdKY9/view?usp=sharing)|[Fire Engine](https://drive.google.com/file/d/1Ff1kvcfMV8vTRaFjebKmlgf2u6OjgFAF/view?usp=sharing)|[Model T](https://drive.google.com/file/d/1bi0FsUtULT73dqm9VAwBjr2Gfbb26Rgj/view?usp=sharing)|[Parachute](https://drive.google.com/file/d/10Z8PDYB_x4L7wGCYdbhvC3F8iUaohr_D/view?usp=sharing)|[Snowmobile](https://drive.google.com/file/d/1T36_K7sC393aMZW8RegnOqINC56RIGuZ/view?usp=sharing)|[Street Sign](https://drive.google.com/file/d/1_V6HuCrgW81l6LvAlDwNTtarmoSpTOAK/view?usp=sharing)|
|Dense121|[Grey Owl](https://drive.google.com/file/d/1a-p4uIWf90oNbzCCeQI_0zMo6giRQaXe/view?usp=sharing)|[Goose](https://drive.google.com/file/d/1QTROVAUJfvoBaeSHSI-Fbsns7hOcsy8P/view?usp=sharing)|[French Bulldog](https://drive.google.com/file/d/1PZcFUKN_Rnq2pA0mwihVvIdOL7WB-nki/view?usp=sharing)|[Hippopotamus](https://drive.google.com/file/d/1ye3y1dsMLOI6BLLoMnEPJnFy8KmgxPHP/view?usp=sharing)|[Cannon](https://drive.google.com/file/d/12YCZ5HdSrDPUHXsAWyVKszwpBh34liLe/view?usp=sharing)|[Fire Engine](https://drive.google.com/file/d/1MGwtQhlFEo8TOa9_ER5LHqx9mduSrddE/view?usp=sharing)|[Model T](https://drive.google.com/file/d/1OqRRNp3yskMmP7cIPYDyfj-jytBA4Yrw/view?usp=sharing)|[Parachute](https://drive.google.com/file/d/1rWP4kjU6TktJYhSuaV2MGzsVV6JeGSKJ/view?usp=sharing)|[Snowmobile](https://drive.google.com/file/d/1kAVH-86h3ytIunT-s-3r0q4LG_U_ZT6R/view?usp=sharing)|[Street Sign](https://drive.google.com/file/d/1-PjlO19HdlVAmyvnW6JQc-IkhJfXZAAw/view?usp=sharing)|


## Installation

```
conda create --name M3D -y python=3.7.0
conda activate M3D
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch

#apex
git clone https://github.com/NVIDIA/apex
cd apex
# if pip >= 23.1 (ref: https://pip.pypa.io/en/stable/news/#v23-1) which supports multiple `--config-settings` with the same key... 
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
# otherwise
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

## Training


Run the following command to train a generator:
```
CUDA_VISIBLE_DEVICES=0,1,2 nohup  python3 -m torch.distributed.launch --nproc_per_node=3 --master_port 23411 train_M3D.py --gs  --match_target 802 --batch_size 16  --epochs 10 --model_type vgg19_bn --log_dir ./checkpoint/vgg19_bn_M3D_3gpu/ --save_dir ./checkpoint/vgg19_bn_M3D_3gpu --apex_train 1 > ./checkpoint/vgg19_bn_M3D_3gpu/output_802.txt
```


## The Implementation Details of Discriminators

Note that since the training data input to D1 and D2 are the same, the two models need to be initialized slightly different to ensure the model discrepancy loss works. 
We simply use a pre-trained model for one discriminator, and a model fine-tuned for one batch using ImageNet training data for another discriminator. We saved the finetuned models in 'pretrain_save_models' for convenience here. You can also finetune the discriminator during the training period！

[vgg19_bn](https://drive.google.com/file/d/1u-emEvbg4d0E4r0WuhcNYPvB729V0YTT/view?usp=sharing)|[resnet50](https://drive.google.com/file/d/1trHp8Hqb_4WHXBoE0YNTJAr5u3eNKm7S/view?usp=sharing)|[densenet121](https://drive.google.com/file/d/1DLbAP_KMNrzCeRkKNWFO6IHfdpwqmGrT/view?usp=sharing)

## Evaluation

 
Run the following command to evaluate transferability of the 10 targets to black-box model on the ImageNet-Val.
```
  CUDA_VISIBLE_DEVICES=0 python eval_M3D.py  --source_model resnet50 --target_model densenet121
```

## Contact
zhaoanqiii@gmail.com

Suggestions and questions are welcome!

