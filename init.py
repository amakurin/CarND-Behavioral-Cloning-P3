import csv
import cv2
import random
import numpy as np
import sklearn
import labutils as lu

def readlog(log_path = './data/driving_log.csv', img_path = './data/IMG/'):
	lines = []
	with open(log_path) as csvlog:
		reader = csv.reader(csvlog)
		for line in reader:
			for i in range(0,3):
				line[i] = img_path + line[i].split('\\')[-1]
			for i in range(3,7):
				line[i] = float(line[i])
			lines.append(line)
	return lines;

def get_sample(log_line, keep_direct_threshold = 0.1, 
			direct_threshold = 0.1, side_angle = 0.2):
	index = 0
	add = 0
	angle = log_line[3] 
	if angle < direct_threshold:
		if np.random.uniform() > keep_direct_threshold:
			# steering additions center, left, right
			diffs = [0, -1.0 * side_angle, side_angle]
			index = np.random.randint(1,3)
			add = diffs[index] 
	img_path = log_line[index]
	img = cv2.imread(img_path)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	angle = angle + add 
	return (img, angle)

def random_flip(img, angle):
	if np.random.randint(0,2)==0:
		img = cv2.flip(img, 1)
		angle = -1.0 * angle
	return img, angle

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
				img, angle = get_sample(batch_sample, 
										keep_direct_threshold=keep_direct_threshold,
										direct_threshold=direct_threshold,
										side_angle=side_angle)
				if flip_random:
					img, angle = random_flip(img, angle)
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
	#model.add(Conv2D(3,1))
	#model.add(Conv2D(16,5))
	#model.add(Activation('relu'))
	#model.add(MaxPooling2D())
	#model.add(Conv2D(32,5))
	#model.add(Activation('relu'))
	#model.add(MaxPooling2D())
	model.add(Flatten())
	#model.add(Dense(512))
	#model.add(Dropout(0.5))
	#model.add(Activation('relu'))
	#model.add(Dense(128))
	#model.add(Dropout(0.5))
	#model.add(Activation('relu'))
	#model.add(Dense(64))
	#model.add(Activation('relu'))
	model.add(Dense(1))
	
	model.compile(loss='mse', optimizer='adam')
	return model

log = readlog()
from sklearn.model_selection import train_test_split
train_log, valid_log = train_test_split(log, test_size=0.2)

new_shape = (32,128,3)
crop_param = ((70, 25), (0, 0))
resize_param = new_shape[:2]
train_generator = generator(train_log, 
							keep_direct_threshold = 0.2, 
							flip_random = True,
							resize_param=resize_param, 
							crop_param=crop_param,
							add_distortion=True,
							randomize_light=True)
valid_generator = generator(valid_log, 
							resize_param=resize_param, 
							crop_param=crop_param)

model = create_model(input_shape= new_shape)
model.fit_generator(train_generator, 
					steps_per_epoch = 500, 
					validation_data=valid_generator, 
					validation_steps=20, epochs=5)

model.save('model.h5')


#img, angle = get_sample(log[0])
#img, angle = random_flip(img, angle)
#
#cv2.imshow('angle {}'.format(angle),img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
#
#img = lu.distort(img)
#cv2.imshow('distort {}'.format(angle),img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
#
#img = lu.randomize_light(img)
#cv2.imshow('light {}'.format(angle),img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
#
#img = lu.crop(img, ((70, 25), (0, 0)))
#cv2.imshow('crop {}'.format(angle),img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
#
#img = lu.resize(img, (32,128))
#cv2.imshow('resize {}'.format(angle),img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
#print (img.shape)