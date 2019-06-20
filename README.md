# Detecting Overfitting of Deep Generators via Latent Recovery

pytorch implementation of "Detecting Overfitting of Deep Generators via Latent Recovery", CVPR, 2019 [cvf link](http://openaccess.thecvf.com/content_CVPR_2019/html/Webster_Detecting_Overfitting_of_Deep_Generative_Networks_via_Latent_Recovery_CVPR_2019_paper.html)

# Demo
Run the demo by calling bash config_latent_recovery_pggan. First download the networks from [this google drive](https://drive.google.com/open?id=11KJXTqo7u_J9E-Pucz5S4tryBU_oPXXr) and place them in a folder named 'networks' (or a directory of your choice). Then modify the path to your CelebA-HQ dataset in config_latent_recovery_pggan. (Note: CelebA-HQ must be a folder of images, compatible with datasets.ImageFolder).

To run latent_recovery.py on your own networks, save the network directly (with torch.save(..)), place the network definition file in this folder (for example DCGAN_ryan.py) and provide the network path / name when calling latent_recovery.

# Dependencies

* pytorch
* scipy
* numpy

# Example Images (see paper)
A common heuristic to detect overfitting is by providing dataset neearest neighbors to generated images. Here, we find the closeset image a generator can produce to a give train or test image, which is more consistent when considering image transformations (see below figure)

 Recovery vs NN in dataset
![](https://i.imgur.com/uW6bPz2.png) 

Finally, networks where overfitting is present also exhibit worse visual results when doing recovery.

Recovery with PGGAN, Mescheder et al, GLO (with 256 training images) and CycleGAN (with 256 training)
![](https://i.imgur.com/XRKRvPW.jpg)
