# CcGAN method

### What is it?

CcGAN (Continious Conditional General Adverserial Network) is a GAN model tweaked to work specifically on continous (regression tasks). For instance when creating new images based on a continous variable such as "age", or "angle". The [paper](https://arxiv.org/abs/2011.07466) implementing CcGAN makes many new novel changes, takign advantage of the continous nature of the data, instead of being limited by it.

### How does it work?


### Why does it fits this task?

In the [csiro-biomass](https://www.kaggle.com/competitions/csiro-biomass) comptetion, datasets are continous, and there are a limited amount of images for such a image regression problem, meaning image augementation or synthesis could be a valid method to increase model performance.

### Implementation

Initially, cloning [the repo](https://github.com/UBCDingXin/improved_CcGAN) and running the code with our dataset proved challening.
The code was hardcoded to work with specific datasets

**Dataset conversion**: I needed to convert the dataset to a `.h5` dataset. Labels needed to match some exact variable names, so goign through the code and changing these to appropriate variabels was required.

**Image differences**: One big issue was the differing image size between the csiro-biomass dataset, and the datasets used by the CcGAN. Upto 192x192 (36864 pixels) images was the maximum used for this task, and so trying to implement creating new 1000x2000 (
2000000 pixels) images would be very difficult.
After talking to the course administrator, we reached an agreement where we would instead resize the 1000x2000 images to 64x64 (4096) and try to run the CcGAN model on these images. This image compression does mean that we now only have 0.2048% of the pixels in the original image.

**Target variable**: The original CcGAN paper only provides code for making new images based on *a single label*, (age, angle.. etc). The csiro-biomass problem is asking us to do regression on *5* labels. This mismatch meant that creating images based on 5 labels would be difficult. In theory it is possible to modify both the **Improved label input** and **Vicinial Risk Minimization** logic to use euclidian distance in a vector space to measure similarity between images, instead of the scalar comparison that is now used. However this is out of scope of this paper, but would be interesting to try out.

**Training time**: Even with this massive image compression, traning and evaluating all the models took 15~ hours in total (64x64 pixels). One can only imagine how much time 1000x2000 images would take.

All preprocessing can be seen in `src/main/ccgan_improved/preprocess/preprocess.py`.

**Old code**: The codebase was last updated 6 months ago, but most of the code was written 5-6 years ago. I was using a newer python and torch version, so many deprecated methods needed to be updated. For example, `astype(np.float)` was used everywhere, but `astype(np.float64)` should now be used instead.
Many comments in the code were hard to understand, and sometimes large code blocks are mysteriously left commented out.



I did a refactor of the codebase, moving files to appropriate locations.

### Why did it not work


### How to run

Firstly, ensure the image2biomass dataset exists in `src/data`.

The code can be run by opening a terminal in the root of the `inf367a_image2biomass` folder and running `sh src/main/ccgan_improved/run_train.sh`