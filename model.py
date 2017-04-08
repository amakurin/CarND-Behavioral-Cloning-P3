import random
import numpy as np
import sklearn
import labutils as lu
from sklearn.model_selection import train_test_split

def train_valid_split(samples):
	return train_test_split(samples, test_size=0.2)

def generator(samples, batch_size=128, 
			keep_direct_threshold = 1., direct_threshold = 0.0005, side_angle = 0.15, 
			flip_random = False, 
			add_distortion = False,
			randomize_light = False,
			resize_param = None, crop_param = None, angle_threshold = 0):
	num_samples = len(samples)
	while 1:
		random.shuffle(samples)
		for offset in range(0, num_samples, batch_size):
			batch_samples = samples[offset:offset+batch_size]
			images = []
			angles = []
			for batch_sample in batch_samples:
				img = None
				while img is None:
					img, angle = lu.get_sample(batch_sample, 
											keep_direct_threshold=keep_direct_threshold,
											direct_threshold=direct_threshold,
											side_angle=side_angle,
											angle_threshold = angle_threshold)
				if flip_random:
					img, angle = lu.random_flip(img, angle)
				if crop_param is not None:
					img = lu.crop(img, crop_param)	
				if add_distortion:
					img, angle = lu.distort(img, angle)
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

def model_v2(input_shape):
	model = Sequential()
	model.add(Lambda(lambda x: x / 127.5 - 1., input_shape=input_shape))
	model.add(Conv2D(3,1))
	
	model.add(Conv2D(24,5, strides=(2, 2)))
	model.add(Activation('relu'))
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

def model_nvid(input_shape):
	model = Sequential()
	model.add(Lambda(lambda x: x / 127.5 - 1., input_shape=input_shape))
	model.add(Conv2D(3,1))
	model.add(Activation('relu'))
	
	model.add(Conv2D(24,5, strides=(2, 2)))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	
	model.add(Conv2D(36,5, strides=(2, 2)))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	
	model.add(Conv2D(48,5, strides=(2, 2)))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	
	model.add(Conv2D(64,3))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	
	model.add(Conv2D(64,3))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	
	model.add(Flatten())
	model.add(Dense(512))
	model.add(Dropout(0.5))
	model.add(Activation('relu'))
	model.add(Dense(128))
	model.add(Dropout(0.5))
	model.add(Activation('relu'))
	model.add(Dense(32))
	model.add(Dropout(0.5))
	model.add(Activation('relu'))
	model.add(Dense(1))
	
	return model

def create_model(version='default', input_shape= (160,320,3)):
	print ('Model used: {}'.format(version))
	model_ctors = { 'nvidia': model_nvid, 
					'v2': model_v2,
					'default': model_v2}
	return model_ctors[version](input_shape)

def compile_model(model):
	model.compile(loss='mse', optimizer='adam')
	return model

def train_model(model_file_name='model.h5',
				log_path = './data/driving_log.csv', 
				img_path = './data/IMG/',
				epochs = 5,
				version = 'default'):
	log = lu.readlog(log_path=log_path, img_path=img_path)
	train_log, valid_log = train_valid_split(log)

	new_shape = (65,320,3)#(66,200,3)
	crop_param = ((70, 25), (0, 0))
	resize_param = new_shape[:2]
	train_generator = generator(train_log, 
								keep_direct_threshold = 0.1, 
								direct_threshold = 0.0005,
								flip_random = True,
								resize_param=resize_param, 
								crop_param=crop_param,
								add_distortion=True,
								randomize_light=True)
	valid_generator = generator(valid_log, 
								resize_param=resize_param, 
								crop_param=crop_param)

	model = create_model(version=version, input_shape= new_shape)
	model = compile_model(model)
	model.fit_generator(train_generator, 
						steps_per_epoch = 300, 
						validation_data=valid_generator, 
						validation_steps=60, epochs=epochs)

	model.save(model_file_name)

def fine_tune_model(src_file_name='model.h5',
					tgt_file_name='model_tuned.h5',
					log_path = './data/driving_log.csv', 
					img_path = './data/IMG/',
					epochs = 5,
					version = 'default',
					angle_threshold = 0):
	log = lu.readlog(log_path=log_path, img_path=img_path)
	train_log, valid_log = train_valid_split(log)

	new_shape = (65,320,3)#(66,200,3)
	crop_param = ((70, 25), (0, 0))
	resize_param = new_shape[:2]
	
	train_generator = generator(train_log, 
								keep_direct_threshold = 0.1, 
								direct_threshold = 0.0005,
								flip_random = True,
								resize_param=resize_param, 
								crop_param=crop_param,
								add_distortion=True,
								randomize_light=True,
								angle_threshold = angle_threshold)
	valid_generator = generator(valid_log, 
								resize_param=resize_param, 
								crop_param=crop_param)
	model = create_model(version=version, input_shape= new_shape)
	model.load_weights(src_file_name)
	model = compile_model(model)
	
	model.fit_generator(train_generator, 
						steps_per_epoch = 300, 
						validation_data=valid_generator, 
						validation_steps=60, epochs=epochs)

	model.save(tgt_file_name)

