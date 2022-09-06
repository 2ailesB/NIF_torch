# NIF_torch
An implementation of Neural Implicit Flow (Pan et al. - 2022) using PyTorch.
Tensorflow version at :  https://github.com/pswpswpsw/nif.  

![NIF architecture](https://github.com/2ailesB/nif_torch/img/NIF.jpg)

## Results
### simple NIF


### multiscale NIF

### last layer NIF
Using the cfg file 'niflastlayer_cylinder' in the config folder, we have the following result :

![Training loss with last layer nif on the cylinder dataset](img/cyl_train_loss.jpg)
![Training results with last layer nif on the cylinder dataset](img/cyl_vis_train.jpg)
![Testing loss with last layer nif on the cylinder dataset](img/cyl_test_loss.jpg)
![Testing results with last layer nif on the cylinder dataset](img/cyl_vis_test.jpg)

## Key References
<a id="1" href="https://arxiv.org/abs/2204.03216">[1]</a> Neural Implicit Flow : a mesh-agnostic dimensionality reduction of spatio-temporal data, Shaowu Pan, Steven L. Brunton, J. Nathan Kutz,  2022  