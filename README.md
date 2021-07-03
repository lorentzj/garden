# Garden (v1)

A spell to create plants out of the aether.

https://twitter.com/garden___bot

## Models

The image generator is a [DCGAN](https://arxiv.org/pdf/1511.06434.pdf), with a couple extra layers for handling the image type data (flower, leaf, and so on). It is trained on photos from [ImageCLEF](http://imageclef.org/2013/plant). The training script is based on [PyTorch's DCGAN tutorial](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html).

The species name generator is a Markov model that predicts the next character based on the previous 4. It is trained on species names from [World Flora Online](http://worldfloraonline.org).

## Install

`git clone https://github.com/lorentzj/garden`

`cd garden`

`conda create -p ./env --file requirements.txt`

`conda activate ./env`

`jupyter notebook`