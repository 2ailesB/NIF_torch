# NIF_torch
An implementation of Neural Implicit Flow (Pan et al. - 2022) using PyTorch.
Tensorflow/keras version at :  https://github.com/pswpswpsw/nif.  

![NIF architecture](https://github.com/2ailesB/NIF_torch/blob/main/img/NIF.png)

## Results
### simple NIF
Using the cfg file 'nif_1dwave.yaml' in the config folder, we have the following result :

|Loss|Visual results|
|:--:|:------------:|
|![Training loss with last layer nif on the cylinder dataset](https://github.com/2ailesB/NIF_torch/blob/main/img/nifs_train_loss.png)|![Training results with last layer nif on the cylinder dataset](https://github.com/2ailesB/NIF_torch/blob/main/img/nifs_vis_train.png)|
|![Testing loss with last layer nif on the cylinder dataset](https://github.com/2ailesB/NIF_torch/blob/main/img/nifs_val_loss.png)|![Testing results with last layer nif on the cylinder dataset](https://github.com/2ailesB/NIF_torch/blob/main/img/nifs_vis_test.png)|


### multiscale NIF
Using the cfg file 'nifmultiscale.yaml' in the config folder, we have the following result :

|Loss|Visual results|
|:--:|:------------:|
|![Training loss with last layer nif on the cylinder dataset](https://github.com/2ailesB/NIF_torch/blob/main/img/nifms_train_loss.png)|![Training results with last layer nif on the cylinder dataset](https://github.com/2ailesB/NIF_torch/blob/main/img/nifms_vis_train.png)|
|![Testing loss with last layer nif on the cylinder dataset](https://github.com/2ailesB/NIF_torch/blob/main/img/nifms_val_loss.png)|![Testing results with last layer nif on the cylinder dataset](https://github.com/2ailesB/NIF_torch/blob/main/img/nifms_vis_test.png)|


### last layer NIF
Using the cfg file 'niflastlayer_cylinder.yaml' in the config folder, we have the following result :

|Loss|Visual results|
|:--:|:------------:|
|![Training loss with last layer nif on the cylinder dataset](https://github.com/2ailesB/NIF_torch/blob/main/img/nifll_train_loss.png)|![Training results with last layer nif on the cylinder dataset](https://github.com/2ailesB/NIF_torch/blob/main/img/nifll_vis_train.png)|
|![Testing loss with last layer nif on the cylinder dataset](https://github.com/2ailesB/NIF_torch/blob/main/img/nifll_val_loss.png)|![Testing results with last layer nif on the cylinder dataset](https://github.com/2ailesB/NIF_torch/blob/main/img/nifll_vis_test.png)|

## Key References
<a id="1" href="https://arxiv.org/abs/2204.03216">[1]</a> Neural Implicit Flow : a mesh-agnostic dimensionality reduction of spatio-temporal data, Shaowu Pan, Steven L. Brunton, J. Nathan Kutz,  2022  