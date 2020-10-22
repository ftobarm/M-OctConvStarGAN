# Tob-StarGAN in PyTorch


--------------------------------------------------------------------------------
This repository provides a PyTorch implementation of a variation of [StarGAN](https://arxiv.org/abs/1711.09020)  that use Multi-Octave Convolution that replace the standar convolution in the discriminator and generator. See [https://github.com/yunjey/stargan](https://github.com/yunjey/stargan) for the original content. 

The 2 diferent implemetation:

* main.py : Original implementation of StarGAN
* main.main_MultiOctConv.py: Implementation of StarGAN with the convolution layer changed by M-OctConv layers.



<br/>



## Dependencies
* [Python 3.5+](https://www.continuum.io/downloads)
* [PyTorch 1.4.0+](http://pytorch.org/)
* [TensorFlow 1.3+](https://www.tensorflow.org/) (optional for tensorboard)
* [scikit-learn 0.20.0](https://scikit-learn.org/stable/index.html)


<br/>

## Usage
to train on CelebA:
```
$ python [main.py | main_Solo_D.py | main_MultiOctConv.py] --mode train --dataset CelebA --image_size 128 --c_dim 5 \
                 --sample_dir stargan_celeba/samples --log_dir stargan_celeba/logs \
                 --model_save_dir stargan_celeba/models --result_dir stargan_celeba/results \
                 --selected_attrs Black_Hair Blond_Hair Brown_Hair Male Young
```

To train on RaFD:

```
$ $ python [main.py | main_Solo_D.py | main_MultiOctConv.py] --mode train --dataset RaFD --image_size 128 --c_dim 8 \
                 --sample_dir stargan_rafd/samples --log_dir stargan_rafd/logs \
                 --model_save_dir stargan_rafd/models --result_dir stargan_rafd/results
```

To train on both CelebA and RafD:

```
$ $ python [main.py  | main_MultiOctConv.py] --mode=train --dataset Both --image_size 256 --c_dim 5 --c2_dim 8 \
                 --sample_dir stargan_both/samples --log_dir stargan_both/logs \
                 --model_save_dir stargan_both/models --result_dir stargan_both/results
```

To train on your own dataset, create a folder structure in the same format as [RaFD](https://github.com/yunjey/StarGAN/blob/master/jpg/RaFD.md) and run the command:

```
$ python [main.py | main_MultiOctConv.py] --mode train --dataset RaFD --rafd_crop_size CROP_SIZE --image_size IMG_SIZE \
                 --c_dim LABEL_DIM --rafd_image_dir TRAIN_IMG_DIR \
                 --sample_dir stargan_custom/samples --log_dir stargan_custom/logs \
                 --model_save_dir stargan_custom/models --result_dir stargan_custom/results
```


### 4. Testing

To test  on CelebA:

```
$ python [main.py | main_Solo_D.py | main_MultiOctConv.py] --mode test --dataset CelebA --image_size 128 --c_dim 5 \
                 --sample_dir stargan_celeba/samples --log_dir stargan_celeba/logs \
                 --model_save_dir stargan_celeba/models --result_dir stargan_celeba/results \
                 --selected_attrs Black_Hair Blond_Hair Brown_Hair Male Young
```

To test  on RaFD:

```
$ python [main.py | main_Solo_D.py | main_MultiOctConv.py] --mode test --dataset RaFD --image_size 128 \
                 --c_dim 8 --rafd_image_dir data/RaFD/test \
                 --sample_dir stargan_rafd/samples --log_dir stargan_rafd/logs \
                 --model_save_dir stargan_rafd/models --result_dir stargan_rafd/results
```

To test  on both CelebA and RaFD:

```
$ python [main.py | main_MultiOctConv.py] --mode test --dataset Both --image_size 256 --c_dim 5 --c2_dim 8 \
                 --sample_dir stargan_both/samples --log_dir stargan_both/logs \
                 --model_save_dir stargan_both/models --result_dir stargan_both/results
```

To test  on your own dataset:

```
$ python [main.py | main_MultiOctConv.py] --mode test --dataset RaFD --rafd_crop_size CROP_SIZE --image_size IMG_SIZE \
                 --c_dim LABEL_DIM --rafd_image_dir TEST_IMG_DIR \
                 --sample_dir stargan_custom/samples --log_dir stargan_custom/logs \
                 --model_save_dir stargan_custom/models --result_dir stargan_custom/results
```
