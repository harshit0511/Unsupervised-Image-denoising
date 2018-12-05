# Unsupervised-Image-denoising
Reference paper - https://arxiv.org/pdf/1803.04189.pdf

Introduction
============

Recent advances in deep neural networks have sparked significant
interest in denoising techniques without using any well-curated
mathematical model to remove noise. Noisy images can be cleaned using
clean images as target images and training a neural network on the
(noisy, clean) pair but as we show in this project, similar results can
be achieved by using (noisy, noisy) pairs as long as the underlying
image remains the same. The loss functions for both methods can be
defined as:

$$Supervised: \sum_i L(f(noisy_i| \theta), clean_i)$$
$$Unsupervised: \sum_i L(f(noisy_{1, i}| \theta), noisy_{2, i})$$ The
image denoiser could have applications in a plethora of domains,
including military and healthcare, where clean target images are not
readily available or take significant amount of human intervention to
denoise them.

Related work
============

-talk about other papers published on similar topic and evaluate them.

Previous techniques have existed where deep neural networks were used to
denoise images [@DBLP:journals/corr/MaoSY16a] but these methods have
relied on clean images for training. Denoising the image using corrupted
target images was performed recently where Lethtinen et al., 2018
[@DBLP:journals/corr/abs-1803-04189] showed that clean images are not
necessary for denoising images, which can be applied to a wide range of
areas. As far as the architecture is concerned, we used a U-net which is
essentially a combination of Convolutional neural networks, pooling,
ReLu activation in a U-shaped architecture
[@DBLP:journals/corr/RonnebergerFB15] with deconvolutional units which
generate the output in the shape of an image and has been widely used
for cases where outputs are required to be images. We used code segments
from an unofficial implementation of the Noise2Noise paper [@yu4u]. We
used the pre-built network but used our own image dataset and
experimented with hyper-parameters to come up with the results.

Problem formulation
===================

elongate using original paper (give some theoretical background) how is
the image getting denoised without seeing actual image (expectation)

The objective is to apply synthetic noise on the image dataset, train
the network, recover the images using the trained graph and observe the
recovery loss. Different noise settings were used for source and target
images but since the underlying image was the same, clean images were
restored. We depreciated the size of the image to 128 x 128 to account
for computing issues which may have been encountered on using too many
features. At the same time, low-resolution images could not be used for
this task as denoising might not be effective.

Methods
=======

U-NET, SRRESNET (screenshot), Bayesian optimization (theory behind it)
add subsection about Bayesian learning

We have used a U-NET to perform Image denoising to remove Gaussian noise
and Textual noise from an image. To do this, we used the IMAGENET
dataset [@imagenet_cvpr09] and trained on 290 images from this dataset.
We trained the network for around 15-20 epochs for one particular level
of noise. We measured our performance using PSNR (Peak Signal Noise
Ratio). For each task we performed, we used a different loss function.
We used $ L_2 $ loss for Gaussian noise removal, $ L_1 $ error for
Textual Noise removal, and $ L_0 $ Loss for Random Impulse noise
removal. $$L_0 : \frac{1}{N} \sum_{i} \lim_{p\to 0} \sqrt[p]{(y_i)^p}$$
$$L_1 : \frac{1}{N} \sum_i |y - y_i|$$
$$L_2 : \frac{1}{N} \sum_i \sqrt{(y -y_i)^2}$$

Noise models
------------

elongate using original paper

[**Gaussian noise**]{} model was obtained by sampling from a gaussian
distribution with a constant mean but differing variances corresponding
to the noise level.\
[**Random Valued Impulse noise**]{} was added using a mask to add on to
the original image. The mask contained values generated from binomial
distribution whose probability came from a uniform distribution. The
maximum occupancy was tried up to 100 where image was almost completely
covered in noise and wasn’t able to recover back to its original state.\
[**Textual noise**]{} was added using Uniform distribution to decide on
the amount of space to be occupied by random texts. The target images in
this case were peppered with similar noise model but since the added
text and the occupancy is randomly allocated, train and test images were
distinct. The maximum number of words in the image was tried up to 75
where much of the information was getting lost.

Architecture and design
=======================

add SRRESNET

We implemented U-Net to train the network in this part, we plan on using
other architectures, specially SRRESNET [@DBLP:journals/corr/LimSKNL17]
in coming days. Figure (1) was generated using Tensorboard after
training the network. The actual graph on tensorboard contained a lot of
nodes as the network was very deep but this image shows a brief overview
of our network. We experimented with different number of hidden layers
and the same graph can be extended for deeper networks as well.
Up-sampling was done by performing deconvolution which was implemented
using Conv2DTranspose in keras.

![U-shaped network used for
training[]{data-label="fig:1"}](Computational_graph)

Results
=======

Add clean target training trend Add screenshots of noise removal

[**GitHub repository**]{} :
[[github.com/harshit0511/Unsupervised-Image-denoising](https://github.com/harshit0511/Unsupervised-Image-denoising)]{}

We experimented by increasing the noise level until the original image
was not recovered. PSNR values of below  13 were obtained for recovered
images that were not denoised properly and were beyond recognition. Fig.
2 shows images which were denoised using our network. Fig. 3 shows the
trend of denoising as we increased the level of noise on input images.
Interestingly, as we kept on increasing the noise, denoising was also
gradually decreasing as depicted by the PSNR values. We would like to
further experiment with hyper-parameters of this network and use some
other well-known architectures and see how they compare with U-Net on
Image denoising task.

![Images on the left are the ground truth image, images in the middle
are the generated noisy images which are used as source and target
images, images on the right were the denoised images generated by the
trained network.
[]{data-label="fig:results"}](gaussian_bird_GT.png "fig:") \[fig:bird1\]

![Images on the left are the ground truth image, images in the middle
are the generated noisy images which are used as source and target
images, images on the right were the denoised images generated by the
trained network.
[]{data-label="fig:results"}](raondom_bird_GT.png "fig:") \[fig:bird3\]

![Images on the left are the ground truth image, images in the middle
are the generated noisy images which are used as source and target
images, images on the right were the denoised images generated by the
trained network. []{data-label="fig:results"}](bird_GT.png "fig:")
\[fig:bird2\]

![Plots depicting the change in PSNR Values on the validation set as we
increased the noise on the images. It was interesting to note that as we
increased the intensity of noise on the images, we got a gradually
decreasing PSNR as opposed to a sudden decline after a certain
threshold.) []{data-label="fig:val"}](gaussian_psnr.png "fig:")
\[fig:val1\]

 

![Plots depicting the change in PSNR Values on the validation set as we
increased the noise on the images. It was interesting to note that as we
increased the intensity of noise on the images, we got a gradually
decreasing PSNR as opposed to a sudden decline after a certain
threshold.) []{data-label="fig:val"}](gaussian_loss.png "fig:")
\[fig:val1\]

 

![Plots depicting the change in PSNR Values on the validation set as we
increased the noise on the images. It was interesting to note that as we
increased the intensity of noise on the images, we got a gradually
decreasing PSNR as opposed to a sudden decline after a certain
threshold.) []{data-label="fig:val"}](random_psnr.png "fig:")
\[fig:val3\]

 

![Plots depicting the change in PSNR Values on the validation set as we
increased the noise on the images. It was interesting to note that as we
increased the intensity of noise on the images, we got a gradually
decreasing PSNR as opposed to a sudden decline after a certain
threshold.) []{data-label="fig:val"}](random_loss.png "fig:")
\[fig:val3\]

-

![Plots depicting the change in PSNR Values on the validation set as we
increased the noise on the images. It was interesting to note that as we
increased the intensity of noise on the images, we got a gradually
decreasing PSNR as opposed to a sudden decline after a certain
threshold.) []{data-label="fig:val"}](textual_psnr.png "fig:")
\[fig:val2\]

 

![Plots depicting the change in PSNR Values on the validation set as we
increased the noise on the images. It was interesting to note that as we
increased the intensity of noise on the images, we got a gradually
decreasing PSNR as opposed to a sudden decline after a certain
threshold.) []{data-label="fig:val"}](textual_loss.png "fig:")
\[fig:val3\]

 

Future Work
===========

Noise distribution has to be known Talk about one of the papers
published

We would like to use another model (SRRESNET) to perform noise removal
for gaussian, random and textual noise. We can then compare the
performance of the two neural networks. We would also like to train the
network for more epochs and for that, we will utilize NYU HPC Cluster
environment. Ideally, we would like to train the network for at least 40
epochs. Finally, we will try to formulate our own network for performing
noise removal and compare the results with both SSRESNET and U-NET.
