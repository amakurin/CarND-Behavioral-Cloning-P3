import csv
import cv2
import numpy as np
import math
import random

def tocv(npshape):
	'''
	Flips first two dims of npshape
	'''
	height, width = npshape
	return (width, height)

def random_flip(img, angle, flip_prob = 0.012):
	'''
	Performs random flip of img if it corresponds to left turn
	flip_prob - probability of flipping
	'''
	if angle < 0:
		if np.random.uniform() < flip_prob:
			img = cv2.flip(img, 1)
			angle = -1.0 * angle
	return (img, angle)

def crop(img, topbottom_leftright):
	'''
	Crops img 
	topbottom_leftright - ((top,bottom),(left,right))
	'''
	top = topbottom_leftright[0][0]
	bottom = img.shape[0]- topbottom_leftright[0][1]
	left = topbottom_leftright[1][0]
	right = img.shape[1] - topbottom_leftright[1][1]
	return img[top:bottom, left:right]

def resize(img, new_shape):
	'''
	Resizes img 
	new_shape - (height, weight)
	'''
	cvshape = tocv(new_shape[:2])
	return cv2.resize(img, cvshape, interpolation=cv2.INTER_AREA)
	
def distort(img, angle, max_rot_degrees = 1, max_dx = 20, max_dy = 20, ang_correction = 0.003):
	'''
	Adds distortion to input img and angle
	img - numpy array (height, weight, chans)
	angle - corresponding angle
	max_rot_degrees - maximum angle of rotation to img, will bechosen uniformely random from (-max_rot_degrees,max_rot_degrees)
	max_dx - maximum horizontal translation in pixels
	ang_correction - correction of angle in radians per pixel of horizontal translation
	max_dy - maximum vertical translation in pixels
	'''
	img_shape = img.shape[:2]
	cvshape = tocv(img_shape)
	dangle = np.random.uniform(-max_rot_degrees, max_rot_degrees, 1)[0]
	dx = np.random.uniform(-max_dx, max_dx, 1)[0]
	dy = np.random.uniform(-max_dy, max_dy, 1)[0]
	rot_center = tuple(np.array(cvshape)/2)
	rot_mat = cv2.getRotationMatrix2D(rot_center, dangle, 1)
	rot_mat[0][2] += dx
	rot_mat[1][2] += dy
	result = cv2.warpAffine(img, rot_mat, cvshape, flags=cv2.INTER_LINEAR)
	return (result, angle + dx*ang_correction)	

def randomize_light(img, const_factor = 0.5):
	'''
	Applies random light adjustment 
	'''
	lab = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
	L,a,b = cv2.split(lab)
	L = L * (const_factor + np.random.uniform())
	L[L>255] = 255
	lab = cv2.merge([np.array(L, dtype = np.uint8),a,b])
	return cv2.cvtColor(lab, cv2.COLOR_Lab2RGB)

def readlog(log_path = './data/driving_log.csv', img_path = './data/IMG/'):
	'''
	Reads log from log_path and changes IMG path to img_path if specified
	'''
	lines = []
	with open(log_path) as csvlog:
		reader = csv.reader(csvlog)
		for line in reader:
			if img_path is not None:
				for i in range(0,3):
					line[i] = img_path + line[i].split('\\')[-1]
			for i in range(3,7):
				line[i] = float(line[i])
			
			lines.append(line)
	return lines;

def log_to_angles(log):
	'''
	Reads angles from provided log
	'''
	angles = [] 
	for line in log:
		angles.append(line[3])
	return np.array(angles, dtype=np.float32)

def read_angles(log_path):
	'''
	Reads angles from log at log_path
	'''
	log = readlog(log_path = log_path, img_path = None)	
	return log_to_angles(log)

def balance_log(log_path, undersampling_stds = 8):
	'''
	Performes log balancing by undersampling frequent angles and oversampling rare angles
	undersampling_stds - threshold of angle frequencies for undersampling, count of std of angle frequency distribution
	'''
	print ('===========================')
	print ('===========================')
	print('Balancing log: {} ...'.format(log_path))
	log = readlog(log_path = log_path, img_path = None)	
	angles = log_to_angles(log)
	print('Log size: {}'.format(len(angles)))
	max_angle=np.max(angles)
	print('max_angle: {} \ {}'.format(max_angle, np.degrees(max_angle)))
	min_angle=np.min(angles)
	
	bins=np.linspace(min_angle,max_angle,400)
	hist,bins = np.histogram(angles, bins=bins)
	max_freq = np.max(hist)
	print('max_freq: {}'.format(max_freq))
	mean_freq = math.ceil(np.mean(hist[hist!=max_freq]))
	print('mean_freq: {}'.format(mean_freq))
	std = math.ceil(np.std(hist[hist!=max_freq]))
	print('std: {}'.format(std))

	result = []
	for n in range(0, len(bins)-1):
		bin_start = bins[n]
		bin_end = bins[n+1]
		binned = [line for line in log if (line[3]>=bin_start) and (line[3]<bin_end)]
		binned_len = len(binned)
		if (binned_len > (mean_freq+std*undersampling_stds)):
			random.shuffle(binned)
			result = result + binned[:mean_freq]
			binned = binned[mean_freq:]
			for i in range(0,undersampling_stds):
				random.shuffle(binned)
				result = result + binned[:std]
				binned = binned[std:]
		elif (binned_len>0) and (binned_len<mean_freq):
			while len(binned) < mean_freq:
				binned = binned + binned
			result = result + binned
		else:
			result = result + binned

	print ('...completed')
	print ('===========================')
	return result
	
def log_thresholding(log_path, low_ang_thre_deg = None, high_ang_thre_deg = None):
	'''
	Removes samples from log if angle is greater than low_ang_thre_deg or less than high_ang_thre_deg 
	'''
	log = readlog(log_path = log_path, img_path = None)	
	angles = log_to_angles(log)
	if low_ang_thre_deg is None:
		low_ang_thre = np.max(angles)
	else:
		low_ang_thre = np.radians(low_ang_thre_deg)
	if high_ang_thre_deg is None:
		high_ang_thre = np.min(angles)
	else:
		high_ang_thre = np.radians(high_ang_thre_deg)
	for line in log:
		if (np.abs(line[3]) > high_ang_thre) and (np.abs(line[3]) < low_ang_thre): 
			yield line

def save_log(log, write_to):
	'''
	Saves provided log to file specified in write_to parameter
	'''
	with open(write_to, 'w', newline='') as new_file:
		writer = csv.writer(new_file, delimiter=',')
		writer.writerows(log)

def get_sample(log_line, keep_direct_threshold = 0.1, 
				direct_threshold = 0.0005, side_angle = 0.17, side_rnd = 0.025):
	'''
	Converts log_line to training sample, selects center, left or right camera image
	keep_direct_threshold - probability to use center image with negligible angle
	direct_threshold - threshold for negligible angles in radians 
	side_angle - absolute angle value in radians to use with left or right camera images   
	side_rnd - max random addition to side angles 
	'''
	angle = log_line[3] 
	index = 0
	add = 0
	if np.abs(angle) < direct_threshold:
		if np.random.uniform() > keep_direct_threshold:
			# steering additions center, left, right
			diffs = [0, side_angle, -1.0 *side_angle]
			index = np.random.randint(1,3)
			add = diffs[index] + np.random.uniform(-side_rnd, side_rnd, 1)[0]
	img_path = log_line[index]
	img = cv2.imread(img_path)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	angle = angle + add 
	return (img, angle)


#log = readlog(log_path = './data2/driving_log.csv',	img_path = './data2/IMG/')	
#img, angle = get_sample(log[np.random.randint(0,len(log))])
#cv2.imshow('angle {}'.format(angle),img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
#
#img, angle = random_flip(img, angle)
#
#cv2.imshow('angle {}'.format(angle),img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
#
#img = crop(img, ((70, 25), (0, 0)))
##cv2.imshow('crop {}'.format(angle),img)
##cv2.waitKey(0)
##cv2.destroyAllWindows()
#
#img, angle = distort(img, angle)
##cv2.imshow('distort {}'.format(angle),img)
##cv2.waitKey(0)
##cv2.destroyAllWindows()
##
#img = randomize_light(img)
#cv2.imshow('light {}'.format(angle),img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()


#img = resize(img, (32,128))
#cv2.imshow('resize {}'.format(angle),img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
#print (img.shape)
