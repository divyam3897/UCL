# Representational Continuity </br> for Unsupervised Continual Learning
This is the *Pytorch Implementation* for the paper Representational Continuity for Unsupervised Continual Learning

**Authors**: [Divyam Madaan](https://dmadaan.com/), [Jaehong Yoon](https://jaehong31.github.io), [Yuanchun Li](http://yuanchun-li.github.io), [Yunxin Liu](https://yunxinliu.github.io), [Sung Ju Hwang](http://sungjuhwang.com/)

## Abstract
<img align="middle" width="700" src="https://github.com/divyam3897/UCL/blob/main/concept.png">

Continual learning (CL) aims to learn a sequence of tasks without forgetting the previously acquired knowledge. However, recent advances in continual learning are restricted to supervised continual learning (SCL) scenarios. Consequently, they are not scalable to real-world applications where the data distribution is often biased and unannotated. In this work, we focus on *unsupervised continual learning (UCL)*, where we learn the feature representations on an unlabelled sequence of tasks and show that the reliance on annotated data is not necessary for continual learning. We conduct a systematic study analyzing the learned feature representations and show that unsupervised visual representations are surprisingly more robust to catastrophic forgetting, consistently achieve better performance, and generalize better to out-of-distribution tasks than SCL. Furthermore, we find that UCL achieves a smoother loss landscape through qualitative analysis of the learned representations and learns meaningful feature representations.
Additionally, we propose Lifelong Unsupervised Mixup (Lump), a simple yet effective technique that leverages the interpolation between the current task and previous tasks' instances to alleviate catastrophic forgetting for unsupervised representations.

__Contribution of this work__
- We attempt to bridge the gap between continual learning and representation learning and tackle the two important problems of continual learning with unlabelled data and representation learning on a sequence of tasks.
- Systematic quantitative analysis show that UCL achieves better performance over SCL with significantly lower catastrophic forgetting on Sequential CIFAR-10, CIFAR-100 and Tiny-ImageNet. Additionally, we evaluate on out of distribution tasks and few-shot continually learning demonstrating the expressive power of unsupervised representations. 
- We provide visualization of the representations and loss landscapes that UCL learns discriminative, human perceptual patterns and achieves a flatter and smoother loss landscape. Furthermore, we propose Lifelong Unsupervised Mixup (Lump) for UCL, which effectively alleviates catastrophic forgetting and provides better qualitative interpretations. 


## Prerequisites
```
$ pip install -r requirements.txt
```

## Run
* __Split CIFAR-10__ experiment with SimSiam
```
$ python main.py --data_dir ../Data/ --log_dir ../logs/ -c configs/simsiam_c10.yaml --ckpt_dir ./checkpoints/cifar10_results/ --hide_progress
```

* __Split CIFAR-100__ experiment with SimSiam

```
$ python main.py --data_dir ../Data/ --log_dir ../logs/ -c configs/simsiam_c100.yaml --ckpt_dir ./checkpoints/cifar100_results/ --hide_progress
```

* __Split Tiny-ImageNet__ experiment with SimSiam

```
$ python main.py --data_dir ../Data/ --log_dir ../logs/ -c configs/simsiam_tinyimagenet.yaml --ckpt_dir ./checkpoints/tinyimagenet_results/ --hide_progress
```

* __Split CIFAR-10__ experiment with BarlowTwins
```
$ python main.py --data_dir ../Data/ --log_dir ../logs/ -c configs/barlow_c10.yaml --ckpt_dir ./checkpoints/cifar10_results/ --hide_progress
```

* __Split CIFAR-100__ experiment with BarlowTwins

```
$ python main.py --data_dir ../Data/ --log_dir ../logs/ -c configs/barlowm_c100.yaml --ckpt_dir ./checkpoints/cifar100_results/ --hide_progress
```

* __Split Tiny-ImageNet__ experiment with BarlowTwins

```
$ python main.py --data_dir ../Data/ --log_dir ../logs/ -c configs/barlowm_tinyimagenet.yaml --ckpt_dir ./checkpoints/tinyimagenet_results/ --hide_progress
```

## Contributing
We'd love to accept your contributions to this project. Please feel free to open an issue, or submit a pull request as necessary. If you have implementations of this repository in other ML frameworks, please reach out so we may highlight them here.

## Acknowledgment
The code is build upon [aimagelab/mammoth](https://github.com/aimagelab/mammoth) and [PatrickHua/SimSiam](https://github.com/PatrickHua/SimSiam)

## Citation
If you found the provided code useful, please cite our work.

```bibtex
@inproceedings{
  madaan2022representational,
  title={Representational Continuity for Unsupervised Continual Learning},
  author={Divyam Madaan and Jaehong Yoon and Yuanchun Li and Yunxin Liu and Sung Ju Hwang},
  booktitle={International Conference on Learning Representations},
  year={2022},
  url={https://openreview.net/forum?id=9Hrka5PA7LW}
}
```
