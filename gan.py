from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
import h5py
import numpy as np
import matplotlib.pyplot as plt

class GAN():

	def __init__(self):
		self.img_rows = 150
		self.img_cols = 150
		self.img_depth = 3
		self.img_shape = (self.img_rows, self.img_cols,self.img_depth)
		
		optimizer = Adam(0.0002, 0.5)
		
		# build and compile the discriminator
		self.discriminator = self.build_discriminator()
		self.discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

		# build and compile the generator
		self.generator = self.build_generator()
		self.generator.compile(loss='binary_crossentropy', optimizer=optimizer)

		# generator takes noise as input and generates images
		z = Input(shape=(100,))
		img = self.generator(z)

		# for the combined model we will only train the generator
		self.discriminator.trainable = False

		# the discriminator takes generated images as input and determines validity
		valid = self.discriminator(img)

		# combined model  (stacked generator and discriminator) takes noise as input => generates images => determines validity
		self.combined = Model(z, valid)
		self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)
		
		
	def build_generator(self):

		noise_shape = (100,)

		model = Sequential()

		model.add(Dense(256, input_shape=noise_shape))
		model.add(LeakyReLU(alpha=0.2))
		model.add(BatchNormalization(momentum=0.8))
		model.add(Dense(512))
		model.add(LeakyReLU(alpha=0.2))
		model.add(BatchNormalization(momentum=0.8))
		model.add(Dense(1024))
		model.add(LeakyReLU(alpha=0.2))
		model.add(BatchNormalization(momentum=0.8))
		model.add(Dense(np.prod(self.img_shape), activation='tanh'))
		model.add(Reshape(self.img_shape))

		model.summary()
		noise = Input(shape=noise_shape)
		img = model(noise)

		return Model(noise, img)

	def build_discriminator(self):

		img_shape = (self.img_rows, self.img_cols,self.img_depth)

		model = Sequential()

		model.add(Flatten(input_shape=img_shape))
		model.add(Dense(512))
		model.add(LeakyReLU(alpha=0.2))
		model.add(Dense(256))
		model.add(LeakyReLU(alpha=0.2))
		model.add(Dense(1, activation='sigmoid'))
		model.summary()

		img = Input(shape=img_shape)
		validity = model(img)
		return Model(img, validity)
		
	def train(self, epochs, batch_size=2, sample_interval=10):

		# load dataset
		(X_train) = self.get_data()

		# rescale -1 to 1
		X_train = (X_train.astype(np.float32) - 127.5) / 127.5
		#X_train = np.expand_dims(X_train, axis=3)
		
		half_batch = int(batch_size / 2)

		for epoch in range(epochs):

			# Discriminator

			# select a random half batch of images
			idx = np.random.randint(0, X_train.shape[0], half_batch)
			imgs = X_train[idx]
			noise = np.random.normal(0, 1, (half_batch, 100))

			# generate a half batch of new images
			gen_imgs = self.generator.predict(noise)

			# train discriminator
			d_loss_real = self.discriminator.train_on_batch(imgs, np.ones((half_batch, 1)))
			d_loss_fake = self.discriminator.train_on_batch(gen_imgs, np.zeros((half_batch, 1)))
			d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

			# Generator

			noise = np.random.normal(0, 1, (batch_size, 100))

			# the generator wants the discriminator to label the generated samples as valid (ones)
			valid_y = np.array([1] * batch_size)

			# train the generator
			g_loss = self.combined.train_on_batch(noise, valid_y)

			# plot the progress
			print ("Epoch %d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))
			
			# if at save interval => save generated image samples
			if epoch % sample_interval == 0:
				self.sample_images(epoch)
				
	def sample_images(self, epoch):
		r, c = 1, 1
		noise = np.random.normal(0, 1, (r * c, 100))
		gen_imgs = self.generator.predict(noise)

		# rescale images 0 - 1
		print(gen_imgs)
		gen_imgs = 0.5 * gen_imgs + 0.5
		print(gen_imgs.shape)

		fig, axs = plt.subplots(r, c)
		
		axs.imshow(gen_imgs[0, :,:,0], cmap='jet')
		axs.axis('off')
		
		print("Generating image")
		fig.savefig("images/generated_%d.png" % epoch)
		plt.close()
		
	def get_data(self):
		f = h5py.File('imageweights.h5', 'r')
		
		keys = list(f.keys())
		print(keys)
		
		images = np.array(f[keys[2]])
		
		print('Image shape:', images.shape)
		
		numImages = 1
		
		return images[:numImages]
		
if __name__ == '__main__':
	gan = GAN()
	gan.train(epochs=1000, batch_size=32, sample_interval=50)
	
	#---Saving the Generator Model---
	# serialize model to JSON
	generator_json = gan.generator.to_json()
	with open("generator_model.json", "w") as json_file:
	    json_file.write(generator_json)
	# serialize weights to HDF5
	gan.generator.save_weights("generator_json.h5")
	print("Saved generator model to disk")

