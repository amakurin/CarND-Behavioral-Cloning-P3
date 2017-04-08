import csv
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import labutils as lu
import matplotlib.gridspec as gridspec

def readlog(log_path,	img_path = ''):
	log = lu.readlog(log_path = log_path, img_path = img_path)	
	lines = [] 
	for line in log:
		lines.append(line[3])
	return np.array(lines, dtype=np.float32)

def print_stat(log, epsilon = 0.0005, max_ang = np.radians(25)):	
	print ('Log size: {}'.format(log.shape[0]))
	hist, bins = np.histogram(log, bins=np.linspace(-max_ang,max_ang,100))
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

t1_path = './sdcdata/dt1/driving_log.csv'
t2_path = './sdcdata/dt2/driving_log.csv'
	
t1_log = readlog(t1_path)	
print ('Track 1 ==================')
print_stat(t1_log)

t2_log = readlog(t2_path)	
print ('Track 2 ==================')
print_stat(t2_log)

t1_std_ang = np.std(t1_log)
t2_std_ang = np.std(t2_log)

fig1 = plt.figure(1,figsize=(9, 6))
fig1.suptitle('Angles Histogram', fontsize=14, fontweight='bold')
gs = gridspec.GridSpec(2, 2, hspace=0.6, wspace=0.2)
axt1max = plt.subplot(gs[0])
axt1std = plt.subplot(gs[1])
axt2max = plt.subplot(gs[2])
axt2std = plt.subplot(gs[3])

axt1max.hist(t1_log, bins=np.linspace(-max_ang,max_ang,200))
axt1max.set_title("Track 1 hole dataset")
axt1max.set_xlabel("Steering angle")
axt1max.set_ylabel("Frequency")

axt1std.hist(t1_log, bins=np.linspace(-t1_std_ang,t1_std_ang,100))
axt1std.set_title("Track 1 one std")
axt1std.set_xlabel("Steering angle")
axt1std.set_ylabel("Frequency")

axt2max.hist(t2_log, bins=np.linspace(-max_ang,max_ang,200))
axt2max.set_title("Track 2 hole dataset")
axt2max.set_xlabel("Steering angle")
axt2max.set_ylabel("Frequency")

axt2std.hist(t2_log, bins=np.linspace(-t2_std_ang,t2_std_ang,100))
axt2std.set_title("Track 2 one std")
axt2std.set_xlabel("Steering angle")
axt2std.set_ylabel("Frequency")
plt.show()


#lu.filter_log(t2_path, 17, './sdcdata/dt2/driving_log_thre17.csv')
#lu.filter_log(t2_path, 20, './sdcdata/dt2/driving_log_thre20.csv')
#lu.filter_log(t2_path, 22, './sdcdata/dt2/driving_log_thre22.csv')
#print("completed")