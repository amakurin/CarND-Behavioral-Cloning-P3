import numpy as np
import math
import matplotlib.pyplot as plt
import labutils as lu
import matplotlib.gridspec as gridspec

	
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
	
t1_log = lu.read_angles(t1_path)	
print ('Track 1 ==================')
print_stat(t1_log)
#plot_hist(t1_log, 'Track 1')

t1_straight = list(lu.log_thresholding(t1_path, low_ang_thre_deg = np.degrees(0.0005)))
#lu.save_log(t1_straight, './sdcdata/dt1/driving_log_s.csv')

t1_tough = lu.log_thresholding(t1_path, high_ang_thre_deg = 15)
#lu.save_log(t1_tough, './sdcdata/dt1/driving_log_t.csv')

t1_balansed = lu.balance_log(t1_path)
angles = lu.log_to_angles(t1_balansed)
print ('Track 1 balanced ==================')
print_stat(angles)
#plot_hist(angles, 'Track 1 balanced')
#lu.save_log(t1_balansed, './sdcdata/dt1/driving_log_b.csv')

t2_log = lu.read_angles(t2_path)	
print ('Track 2 ==================')
print_stat(t2_log)
#plot_hist(t2_log, 'Track 2')

t2_straight = lu.log_thresholding(t2_path, low_ang_thre_deg = np.degrees(0.0005))
#lu.save_log(t2_straight, './sdcdata/dt2/driving_log_s.csv')

t2_tough = lu.log_thresholding(t2_path, high_ang_thre_deg = 15)
#lu.save_log(t2_tough, './sdcdata/dt2/driving_log_t.csv')
angles = lu.log_to_angles(t2_tough)
print ('Track 2 tough ==================')
print_stat(angles)


t2_balansed = lu.balance_log(t2_path, undersampling_stds=20)
angles = lu.log_to_angles(t2_balansed)
print ('Track 2 balanced ==================')
print_stat(angles)
#plot_hist(angles, 'Track 2 balanced')
#lu.save_log(t2_balansed, './sdcdata/dt2/driving_log_b.csv')

 
#lu.filter_log(t2_path, 20, './sdcdata/dt2/driving_log_thre20.csv')
#lu.filter_log(t2_path, 22, './sdcdata/dt2/driving_log_thre22.csv')
#print("completed")