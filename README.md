# traditional-decor
This repository is to generate traditional decor patterns.


With this architecture, what we are aiming to do is to generate traditional decor patterns, which are basically patterns that are put in cloth and cups and other such homely utilities. Using data of existing traditional decor patterns, around 450 images, the discriminator learns the patterns and knows what is fake, and the next step is to train the generator to generate its own patterns provided some distribution of noise. At the end of this training, the generator will generate some patterns which will be original and purely generated. Although this might not be very pleasing to watch due to the current size of the dataset, we are looking to add more decor patterns, and generate clearer and more appealing decor patterns.

The dataset is obtained from kaggle, and is in the link https://www.kaggle.com/olgabelitskaya/traditional-decor-patterns, and we are retrieving the h5 file and generating the image from training the generator.

The GAN can be run by running gan.py with python 3.6 with keras installed. 
