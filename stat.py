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

epsilon = 0.0005
t1_path = './sdcdata/dt1/driving_log.csv'
t2_path = './sdcdata/dt2/driving_log.csv'
	
t1_log = readlog(t1_path)	
print ('Track 1 ==================')
t1_size = t1_log.shape[0]
print ('Log size: {}'.format(t1_size))
t1_negligible_freq = t1_log[np.abs(t1_log)<epsilon].shape[0] 
print ('Negligible angles freq: {}'.format(t1_negligible_freq))
t1_valuable_angles = t1_log[np.abs(t1_log)>epsilon]
t1_val_freq = t1_valuable_angles.shape[0] 
t1_left_freq = t1_valuable_angles[t1_valuable_angles<0].shape[0]
t1_right_freq = t1_valuable_angles[t1_valuable_angles>0].shape[0]
print ('Valuable LEFT angles freq: {} \ {}'.format(t1_left_freq,t1_left_freq/t1_val_freq))
print ('Valuable RIGHT angles freq: {} \ {}'.format(t1_right_freq,t1_right_freq/t1_val_freq))

t2_log = readlog(t2_path)	
print ('Track 2 ==================')
t2_size = t2_log.shape[0]
print ('Log size: {}'.format(t2_size))
t2_negligible_freq = t2_log[np.abs(t2_log)<epsilon].shape[0] 
print ('Negligible angles freq: {}'.format(t2_negligible_freq))
t2_valuable_angles = t2_log[np.abs(t2_log)>epsilon]
t2_val_freq = t2_valuable_angles.shape[0] 
t2_left_freq = t2_valuable_angles[t2_valuable_angles<0].shape[0]
t2_right_freq = t2_valuable_angles[t2_valuable_angles>0].shape[0]
print ('Valuable LEFT angles freq: {} \ {}'.format(t2_left_freq,t2_left_freq/t2_val_freq))
print ('Valuable RIGHT angles freq: {} \ {}'.format(t2_right_freq,t2_right_freq/t2_val_freq))

max_ang = np.radians(25)
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


#lu.filter_log(t2_path, 17, './sdcdata/dt2/driving_log_thre.csv')