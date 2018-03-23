# pokemon-gan
Generating Pokemon with a Generative Adversarial Network

## Overview
This project is inspired by **Siraj Raval** in his awesome [video](https://youtu.be/yz6dNf7X7SA).  
The code is inspired by [Newmu](https://github.com/Newmu/dcgan_code), [kvpratama](https://github.com/kvpratama/gan/tree/master/pokemon), [moxiegushi](https://github.com/moxiegushi/pokeGAN) along with my own implemetation.

Training process is very **long**! Please do not train on your little tiny laptop! I have used `Floydhub` to train my model. However, only 2 hours of free trial for GPU option. You can select your favorite cloud computing platform ro train this awesome project, if you find `Training` is interesting.

## Dependencies

```bash
git clone https://github.com/Zhenye-Na/pokemon-gan.git
cd pokemon-gan  
pip install -r requirements.txt
```

> scikit-image  
> tensorflow  
> scipy  
> numpy  
> Pillow


## Usage

```bash
git clone https://github.com/Zhenye-Na/pokemon-gan.git
cd pokemon-gan
python3 main.py
```

## Generative Adversarial Network (GAN)
GAN consist of two network:

 - A discriminator D receive input from training data and generated data. Its job is to learn how to distinguish between these two inputs.
 - A generator G generate samples from a random noise Z. Generator objective is to generate sample that is as real as possible it could not be distinguished by Discriminator.

### Deep Convolution GAN (DCGAN)
In DCGAN architecture, the discriminator `D` is Convolutional Neural Networks (CNN) that applies a lot of filters to extract various features from an image. The discriminator network will be trained to discriminate between the original and generated image. The process of convolution is shown in the illustration below:
<p align="center">
![](http://deeplearning.net/software/theano_versions/dev/_images/same_padding_no_strides_transposed.gif)
</p>

The network structure for the discriminator is given by:
<center>

| Layer        | Shape           | Activation           |
| ------------- |:-------------:|:-------------:|
| input     | batch size, 3, 64, 64 | |
| convolution      | batch size, 64, 32, 32  | LRelu |
| convolution      | batch size, 128, 16, 16  |LRelu | 
| convolution      | batch size, 256, 8, 8  | LRelu |
| convolution      | batch size, 512, 4, 4 | LRelu |
| dense      | batch size, 64, 32, 32 | Sigmoid |

</center>

The generator `G`, which is trained to generate image to fool the discriminator, is trained to generate image from a random input. In DCGAN architecture, the generator is represented by convolution networks that upsample the input. The goal is to process the small input and make an output that is bigger than the input. It works by expanding the input to have zero in-between and then do the convolution process over this expanded area. The convolution over this area will result in larger input for the next layer. The process of upsampling is shown below: 
<p align="center">
![](http://deeplearning.net/software/theano_versions/dev/_images/padding_strides_transposed.gif)
</p>
There are many name for this upsample process: full convolution, in-network upsampling, fractionally-strided convolution, deconvolution, or transposed convolution. 

The network structure for the generator is given by:

<center>

| Layer        | Shape           | Activation           |
| ------------- |:-------------:|:-------------:|
| input     | batch size, 100 (Noise from uniform distribution) | |
| reshape layer      | batch size, 100, 1, 1  | Relu |
| deconvolution      | batch size, 512, 4, 4   |Relu | 
| deconvolution      | batch size, 256, 8, 8  | Relu |
| deconvolution      | batch size, 128, 16, 16 | Relu |
| deconvolution      | batch size, 64, 32, 32 | Relu |
| deconvolution      | batch size, 3, 64, 64 | Tanh |

</center>

### Hyperparameter of DCGAN
The hyperparameter for DCGAN architecture is given in the table below:

<center>

| Hyperparameter        |
| ------------- |
| Mini-batch size of 64     |
| Weight initialize from normal distribution with std = 0.02      |  
| LRelu slope = 0.2      |
| Adam Optimizer with learning rate = 0.0002 and momentum = 0.5      |

</center>

## Pokemon Image Dataset
The dataset of pokemon images are gathered from various sources:

* https://www.kaggle.com/dollarakshay/pokemon-images/discussion
* https://veekun.com/dex/downloads

## Training result
As I have only two-hour GPU option in my account! So I have only get three images of new Pokemons.

![]()
![]()
![]()


## References
* https://www.oreilly.com/learning/generative-adversarial-networks-for-beginners
* https://www.analyticsvidhya.com/blog/2017/06/introductory-generative-adversarial-networks-gans/
* https://github.com/uclaacmai/Generative-Adversarial-Network-Tutorial
* http://blog.aylien.com/introduction-generative-adversarial-networks-code-tensorflow/
* https://www.slideshare.net/ThomasDaSilvaPaula/a-very-gentle-introduction-to-generative-adversarial-networks-aka-gans-71614428
* https://medium.com/@devnag/generative-adversarial-networks-gans-in-50-lines-of-code-pytorch-e81b79659e3f
* https://medium.com/@awjuliani/generative-adversarial-networks-explained-with-a-classic-spongebob-squarepants-episode-54deab2fce39
