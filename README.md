# Generating Photorealistic Images from Sketches using GANs
In this project, we experiment with using GAN models to generate photorealistic images from simple black-and-white sketches. The pix2pix and CycleGAN papers (described below) inspired us to use GAN models to approach this problem.  We tried both a conditional GAN and a CycleGAN model to see how the results would compare.  We were able to achieve **___ accuracy/loss blah blah numbers** when generating these images.

**VIDEO GOES HERE**

**COLAB GOES HERE**

**PRESENTATION GOES HERE**

# Introduction
One initial source of inspiration for this problem was this fun [boredpanda article](https://www.boredpanda.com/kid-drawings-things-i-have-drawn-dom/?utm_source=google&utm_medium=organic&utm_campaign=organic) describing how a dad uses photoshop to turn his son's doodles into photo-realistic images.  We thought it would be interesting to see if we could get similar results using deep learning. In addition to getting cool results, the ability to turn simple line sketches into photo-realistic images could potentially help artists create quick brainstorming images, or help people ideate by sketching instead of typing.

# Related Work
We referenced the [Pix2Pix](https://arxiv.org/pdf/1611.07004.pdf) and [CycleGAN](https://arxiv.org/pdf/1703.10593.pdf) papers for this project. Pix2Pix uses conditional GANs to train on pairs of sketches and photos, where the generator learns to output fake photos as the corresponding image to a given sketch.  The second paper uses a Cycle GAN to train on unpaired data, with no information matching the two categories of photos and sketches.  One apparent difference between our project and the Pix2Pix paper is that while our data is paired, there is more variation between the sketches and photos (i.e. our sketches are hand-drawn, not simply an edge detection or tracing).

We also looked at the [Sketchy](https://www.cc.gatech.edu/~hays/tmp/sketchy-database.pdf) and [SketchMan](https://dl.acm.org/doi/abs/10.1145/3394171.3413720) papers, since initially we wanted to do sketch enhancement.  The Sketchy project (which is where we got our dataset from) deals with mapping handdrawn sketches to photos by matching pose and category, with a goal of supporting image retrieval.  The recent SketchMan paper deals with generating professional-looking, colored illustrations from simple black-and-white sketches, using a complex process of different pipelines to perform this sketch enhancement.

# Approach

## Data
Initially, we planned to generate high-quality colored sketches from low-quality black-and-white sketches. As we did more research, we discovered that acquiring data for this purpose would be time-consuming and we wanted to focus more on the training and models for this project rather than data collection. Because of this, we opted to use the Sketchy database to generate photorealistic images from black-and-white sketches.
We used the Sketchy database, which contains photos of objects from different categories, along with black-and-white rough sketches corresponding to the photos. Some examples of categories include airplane, horse, and wheelchair. Although this database seemed to be a manageable size, we ran into a number of problems trying to use it on Google Colab. The database contained over 400,000 files, which maxed out the file limit for Team Drives. While troubleshooting this issue, we realized that the dataset actually contained several transformed versions of the same files. With this in mind, we opted to eliminate the transformed versions of the files, transforming images as part of the data loading process instead.  This allowed us to upload the data into a Shared Drive for use with Google Colab.

## Models
We chose to try a pix2pix (conditional GAN) model and a CycleGAN model, and compare results from both.  We learned from the CycleGAN and pix2pix papers that CycleGAN models are supposed to work well for unpaired data, while the pix2pix model depends on having paired data. Since our data was paired between photos and sketches, we were interested to try both types of models and see the differences in their results.

The model architecture for the pix2pix cGAN was based off of existing literature focusing on the use of a DCGAN in [Radford and Metz et al. 2016](https://arxiv.org/pdf/1511.06434.pdf). The architecture follows an encoder/decoder fashion, and in the pix2pix model, their "U-Net" architecture contains skip-layers to avoid the typical bottleneck in an encoder/decoder architecture that allows the input to pass information to other layers not directly connected. For example, the first input layer and last output layer may contain information about things such as edges that are not evident when passed through the model directly. While a GAN learns a generative model of the data, a cGAN learns a conditional generative model of the data.

![The pix2pix DCGAN architecture is shown here](results/pix2pix-arch.png)

The original architecture takes in a 100 dimensional distribution, reshapes it into a 4x4x1024 to 8x8x512 using a 4x4 convolution of stride 2, 8x8x512 to 16x16x256 using a 5x5 using a convolution of stride 2, 16x16x256 to 32x32x128 using a convolution of stride 2, then 32x32x128 to 64x64x3 using a convolution of stride 2.

The model architecture for the CycleGAN was based off of existing literature focused on using GANs for style transfer in [Johnson et al. 2016](https://arxiv.org/pdf/1603.08155.pdf). Specifically, the current CycleGAN architecture contains three convolutions, several residual blocks, two fractionally-strided convolutions with stride of 0.5, and one convolution for mapping features to RGB. In addition, there are 6 blocks for 128x128 images and 9 blocks for 256x256 images. We adapted this architecture to work well with the 256x256 images in the Sketchy database.

We hypothesized that training a GAN on a wide variety of types of images might cause it to struggle to learn any one type of image well. In order to test this hypothesis, we experimented with models using the full dataset (across all of the categories) and an individual model for just a few categories, with hopes that for an individual category, the model would hone in on a specific image type and image shape.

## Initial Testing
First we wanted to get a feel for what kind of images could be realistically generated using our dataset. To this end, we made a simple DCGAN based on a tutorial in the Pytorch documentation, which generates random images from the photos in our data. The sketches are not considered in this model, and only the photos were used.

### Results
![The result of training the DCGAN for a low number of iterations](results/DCGAN_low_iterations.png)

In our initial tests, we noticed the generated images exhibited a grid-like pattern. We increased the number of iterations in order to see if this pattern would go away.

![A side-by-side comparison of real images with fake generated images](results/Final_DCGAN_results.png)

After increasing the number of epochs, the grid pattern was no longer visible, and we were able to generate much better-looking random images.

![Chart of loss per iteration for DCGAN](results/DCGAN_loss.png)

Curiously, we noticed that loss did not seem to change much between 1000 and 7000 iterations, despite the significant improvement in quality of generated images.

## Pix2Pix model

In our initial results, our discriminator loss kept converging to 0, preventing the generator from learning. In order to counteract this, we increased our kernel size and added noise to the labels. We also experimented with training the generator twice as much as the discriminator, so that it would have time to gain an advantage. These approaches helped slow down the rate at which the discriminator loss converged to 0, or caused the loss to hover around 0.200. However, unfortunately this did not seem to make much of a difference for training overall.

### Results

<img src="results/pix2pix_owl.png" width="250">
<img src="results/pix2pix_glasses.png" width="250">
It seemed like the pix2pix model didn't learn colors very well, as the results are largely just a modified version of the sketch.

## CycleGAN model

With both of our models, we had an issue where our image results were all black.  We hypothesized that perhaps our model wasn't learning anything because of the large variation in the dataset, or that it was having trouble mapping between black-and-white and colored input.  However, while running the CycleGAN model, we realized that the output values had to be scaled by 255 in order to get the image result to show up with colors.  

### Results

<img src="results/cyclegan_car.png" width="250">
<img src="results/cyclegan_house.png" width="250">
<img src="results/cyclegan_goose.png" width="250">
Although colors and basic form showed up in the output images, the results are far from photorealistic.  The colors are pretty blown-out and saturated.  With more training time, we could probably achieve better results with the model.  However, the CycleGAN still seems to give better results than the Pix2pix model, since the images have more color and seem to be more influenced by training with the photos.

# Discussion
In conclusion, we realized that we may have overestimated our goals given our time constraints and limited experience with GANs.  We learned that GANs are tricky to train and troubleshoot, and that data collection and parsing is a really important and time-consuming part of the project.  We were excited to try a few different models on our data and compare them, but we were disappointed with our lack of reasonable results.  If we did this project again, we would definitely make some changes such as choosing a more targeted dataset (fewer categories and more examples with slight variation) or just try training a single model in order to focus our project and allow more time for debugging and troubleshooting.  We think that this approach would at least give us a clearer path for our project, even if it wouldn't specifically solve the problems we encountered here.
