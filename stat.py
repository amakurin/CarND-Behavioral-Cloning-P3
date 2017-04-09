import csv
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import labutils as lu
import matplotlib.gridspec as gridspec
import random

def log_to_angles(log):
	angles = [] 
	for line in log:
		angles.append(line[3])
	return np.array(angles, dtype=np.float32)

def read_angles(log_path,	img_path = ''):
	log = lu.readlog(log_path = log_path, img_path = img_path)	
	return log_to_angles(log)

def balance_log(log_path, undersampling_stds = 8):
	print ('===========================')
	print ('===========================')
	print('Balancing log: {} ...'.format(log_path))
	log = lu.readlog(log_path = log_path, img_path = None)	
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
	
def print_stat(log, epsilon = 0.0005):	
	print ('Log size: {}'.format(log.shape[0]))
	hist, bins = np.histogram(log, bins=np.linspace(np.min(log),np.max(log),100))
	print('Angles median freq: {}'.format(np.median(hist)))
	print('Angles mean freq: {}'.format(np.mean(hist)))
	print('Angles max freq: {}'.format(np.max(hist)))
	negligible_freq = log[np.abs(log)<epsilon].shape[0] 
	print ('Negligible angles freq: {}'.format(negligible_freq))
	print('Suggested direct rate: {}'.format((np.mean(hist)+np.median(hist))/2/negligible_freq))
	valuable_angles = log[np.abs(log)>epsilon]
	val_freq = valuable_angles.shape[0] 
	left_freq = valuable_angles[valuable_angles<0].shape[0]
	right_freq = valuable_angles[valuable_angles>0].shape[0]
	print ('Valuable LEFT angles freq: {} \ {}'.format(left_freq,left_freq/val_freq))
	print ('Valuable RIGHT angles freq: {} \ {}'.format(right_freq,right_freq/val_freq))
	print('Suggested flip random: {}'.format((left_freq-right_freq)/2/val_freq))

def plot_hist(log, title=''):
	std_ang = np.std(log)
	fig1 = plt.figure(1,figsize=(9, 4))
	fig1.suptitle('Angles Histogram: '+title, fontsize=14, fontweight='bold')
	gs = gridspec.GridSpec(1, 2, hspace=0.6, wspace=0.2)
	axt1max = plt.subplot(gs[0])
	axt1std = plt.subplot(gs[1])

	axt1max.hist(log, bins=np.linspace(np.min(log),np.max(log),200))
	axt1max.set_title("Whole dataset")
	axt1max.set_xlabel("Steering angle")
	axt1max.set_ylabel("Frequency")

	axt1std.hist(log, bins=np.linspace(-std_ang,std_ang,200))
	axt1std.set_title("One std")
	axt1std.set_xlabel("Steering angle")
	axt1std.set_ylabel("Frequency")
	plt.show()

t1_path = './sdcdata/dt1/driving_log.csv'
t2_path = './sdcdata/dt2/driving_log.csv'
	
t1_log = read_angles(t1_path)	
print ('Track 1 ==================')
print_stat(t1_log)
plot_hist(t1_log, 'Track 1')

t1_balansed = balance_log(t1_path)
angles = log_to_angles(t1_balansed)
print ('Track 1 balanced ==================')
print_stat(angles)
plot_hist(angles, 'Track 1 balanced')
lu.save_log(t1_balansed, './sdcdata/dt1/driving_log_b.csv')

t2_log = read_angles(t2_path)	
print ('Track 2 ==================')
print_stat(t2_log)
plot_hist(t2_log, 'Track 2')

t2_balansed = balance_log(t2_path, undersampling_stds=20)
angles = log_to_angles(t2_balansed)
print ('Track 2 balanced ==================')
print_stat(angles)
plot_hist(angles, 'Track 2 balanced')
lu.save_log(t2_balansed, './sdcdata/dt2/driving_log_b.csv')

#lu.filter_log(t2_path, 20, './sdcdata/dt2/driving_log_thre20.csv')
#lu.filter_log(t2_path, 22, './sdcdata/dt2/driving_log_thre22.csv')
#print("completed")