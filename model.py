import random
import numpy as np
import sklearn
import labutils as lu
from sklearn.model_selection import train_test_split

def train_valid_split(samples):
	return train_test_split(samples, test_size=0.2)

def generator(samples, batch_size=128, 
			keep_direct_threshold = 1., direct_threshold = 0.1, side_angle = 0.2, 
			flip_random = False, 
			add_distortion = False,
			randomize_light = False,
			resize_param = None, crop_param = None):
	num_samples = len(samples)
	while 1:
		random.shuffle(samples)
		for offset in range(0, num_samples, batch_size):
			batch_samples = samples[offset:offset+batch_size]
			images = []
			angles = []
			for batch_sample in batch_samples:
				img, angle = lu.get_sample(batch_sample, 
										keep_direct_threshold=keep_direct_threshold,
										direct_threshold=direct_threshold,
										side_angle=side_angle)
				if flip_random:
					img, angle = lu.random_flip(img, angle)
				if crop_param is not None:
					img = lu.crop(img, crop_param)	
				if add_distortion:
					img = lu.distort(img)
				if randomize_light:
					img = lu.randomize_light(img)
				if resize_param is not None:
					img = lu.resize(img, resize_param)
				images.append(img)
				angles.append(angle)
			X_train = np.array(images)
			y_train = np.array(angles)
			yield sklearn.utils.shuffle(X_train, y_train)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D, MaxPooling2D, Dropout, Activation

def create_model(input_shape= (160,320,3)):
	model = Sequential()
	model.add(Lambda(lambda x: x / 127.5 - 1., input_shape=input_shape))
	model.add(Conv2D(3,1))
	
	model.add(Conv2D(24,5, strides=(2, 2)))
	model.add(Activation('relu'))
	#model.add(MaxPooling2D())
	model.add(Dropout(0.5))
	
	model.add(Conv2D(36,5, strides=(2, 2)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D())
	model.add(Dropout(0.5))
	
	model.add(Conv2D(48,3))
	model.add(Activation('relu'))
	model.add(MaxPooling2D())
	model.add(Dropout(0.5))
	
	model.add(Flatten())
	model.add(Dense(512))
	model.add(Dropout(0.5))
	model.add(Activation('relu'))
	model.add(Dense(128))
	model.add(Dropout(0.5))
	model.add(Activation('relu'))
	model.add(Dense(16))
	model.add(Activation('relu'))
	model.add(Dense(1))
	
	return model

def compile_model(model):
	model.compile(loss='mse', optimizer='adam')
	return model

def train_model(model_file_name='model.h5',
				log_path = './data/driving_log.csv', 
				img_path = './data/IMG/'):
	log = lu.readlog(log_path=log_path, img_path=img_path)
	train_log, valid_log = train_valid_split(log)

	new_shape = (65,320,3)#(32,128,3)
	crop_param = ((70, 25), (0, 0))
	resize_param = new_shape[:2]
	train_generator = generator(train_log, 
								keep_direct_threshold = 0.5, 
								flip_random = True,
								#resize_param=resize_param, 
								crop_param=crop_param,
								add_distortion=True,
								randomize_light=True)
	valid_generator = generator(valid_log, 
								#resize_param=resize_param, 
								crop_param=crop_param)

	model = create_model(input_shape= new_shape)
	model = compile_model(model)
	model.fit_generator(train_generator, 
						steps_per_epoch = 500, 
						validation_data=valid_generator, 
						validation_steps=20, epochs=5)

	model.save(model_file_name)

