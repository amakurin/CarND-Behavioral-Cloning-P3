import csv
import cv2
import numpy as np
import math

def tocv(npshape):
	height, width = npshape
	return (width, height)

def random_flip(img, angle, flip_prob = 0.012):
	if angle < 0:
		if np.random.uniform() < flip_prob:
			img = cv2.flip(img, 1)
			angle = -1.0 * angle
	return (img, angle)

def crop(img, topbottom_leftright):
	top = topbottom_leftright[0][0]
	bottom = img.shape[0]- topbottom_leftright[0][1]
	left = topbottom_leftright[1][0]
	right = img.shape[1] - topbottom_leftright[1][1]
	return img[top:bottom, left:right]

def resize(img, new_shape):
	cvshape = tocv(new_shape[:2])
	return cv2.resize(img, cvshape, interpolation=cv2.INTER_AREA)
	
def distort(img, angle, max_rot_degrees = 1, max_dx = 20, max_dy = 20, ang_correction = 0.075):
	'''
	Adds distortion to input img
	img - numpy array (height, weight, chans)
	maxAngle - maximum absolute angle, will be used to select random from [-maxAngle,maxAngle] uniformely
	maxCenterDistance - maximum distance to move center of rotation
	maxScale - maximum scale
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

def log_thresholding(log, ang_threshold_degrees):
	ang_thre = np.radians(ang_threshold_degrees)
	for line in log:
		if np.abs(line[3]) > ang_thre: yield line

def filter_log(log_path, ang_threshold, write_to=None):
	log = readlog(log_path=log_path, img_path=None)
	thresholded = log_thresholding(log, ang_threshold)
	if write_to is not None:
		with open(write_to, 'w', newline='') as new_file:
			writer = csv.writer(new_file, delimiter=',')
			writer.writerows(thresholded)
	return thresholded

def get_sample(log_line, keep_direct_threshold = 0.1, 
				direct_threshold = 0.0005, side_angle = 0.15, side_rnd = 0.025):
	angle = log_line[3] 
	index = 0
	add = 0
#	if np.abs(angle) < direct_threshold:
#		if np.random.uniform() > keep_direct_threshold:
#			# steering additions center, left, right
#			diffs = [0, side_angle, -1.0 *side_angle]
#			index = np.random.randint(1,3)
#			add = diffs[index] + np.random.uniform(-side_rnd, side_rnd, 1)[0]
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
