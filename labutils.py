import csv
import cv2
import numpy as np
import math

def tocv(npshape):
	height, width = npshape
	return (width, height)

def random_flip(img, angle):
	if np.random.randint(0,2)==0:
		img = cv2.flip(img, 1)
		angle = -1.0 * angle
	return img, angle

def crop(img, topbottom_leftright):
	top = topbottom_leftright[0][0]
	bottom = img.shape[0]- topbottom_leftright[0][1]
	left = topbottom_leftright[1][0]
	right = img.shape[1] - topbottom_leftright[1][1]
	return img[top:bottom, left:right]

def resize(img, new_shape):
	cvshape = tocv(new_shape[:2])
	return cv2.resize(img, cvshape, interpolation=cv2.INTER_AREA)
	
def distort(img, maxAngle = np.pi/18, maxCenterDistance = 6, maxScale = 1.2):
	'''
	Adds distortion to input img
	img - numpy array (height, weight, chans)
	maxAngle - maximum absolute angle, will be used to select random from [-maxAngle,maxAngle] uniformely
	maxCenterDistance - maximum distance to move center of rotation
	maxScale - maximum scale
	'''
	img_shape = img.shape[:2]
	cvshape = tocv(img_shape)
	dangle = np.random.uniform(-maxAngle, maxAngle, 1)[0]
	distance = math.sqrt(maxCenterDistance)
	dx = math.floor(np.random.uniform(-distance, distance, 1)[0])
	dy = math.floor(np.random.uniform(-distance, distance, 1)[0])
	dscale = np.random.uniform(2-maxScale, maxScale, 1)[0]
	dcenter = np.array(cvshape)/2 +[dx,dy]
	rot_mat = cv2.getRotationMatrix2D(tuple(dcenter), dangle, dscale)
	result = cv2.warpAffine(img, rot_mat, cvshape, flags=cv2.INTER_LINEAR)
	return result	

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
			diffs = [0, side_angle, -1.0 *side_angle]
			index = np.random.randint(1,3)
			add = diffs[index] 
	img_path = log_line[index]
	img = cv2.imread(img_path)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	angle = angle + add 
	return (img, angle)

#log = readlog()	
#img, angle = get_sample(log[0])
#img, angle = random_flip(img, angle)
#
#cv2.imshow('angle {}'.format(angle),img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
#
#img = distort(img)
#cv2.imshow('distort {}'.format(angle),img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
#
#img = randomize_light(img)
#cv2.imshow('light {}'.format(angle),img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
#
#img = crop(img, ((70, 25), (0, 0)))
#cv2.imshow('crop {}'.format(angle),img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
#
#img = resize(img, (32,128))
#cv2.imshow('resize {}'.format(angle),img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
#print (img.shape)
