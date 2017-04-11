import numpy as np
import math
import matplotlib.pyplot as plt
import labutils as lu
import matplotlib.gridspec as gridspec
import random
	
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
	
#t1_log = lu.read_angles(t1_path)	
#print ('Track 1 ==================')
#print_stat(t1_log)
#plot_hist(t1_log, 'Track 1')
#
#t1_straight = list(lu.log_thresholding(t1_path, low_ang_thre_deg = np.degrees(0.0005)))
#lu.save_log(t1_straight, './sdcdata/dt1/driving_log_s.csv')
#
#t1_tough = lu.log_thresholding(t1_path, high_ang_thre_deg = 15)
##lu.save_log(t1_tough, './sdcdata/dt1/driving_log_t.csv')
#
#t1_balansed = lu.balance_log(t1_path)
#angles = lu.log_to_angles(t1_balansed)
#print ('Track 1 balanced ==================')
#print_stat(angles)
#plot_hist(angles, 'Track 1 balanced')
##lu.save_log(t1_balansed, './sdcdata/dt1/driving_log_b.csv')
#
#t2_log = lu.read_angles(t2_path)	
#print ('Track 2 ==================')
#print_stat(t2_log)
#plot_hist(t2_log, 'Track 2')
#
#t2_straight = lu.log_thresholding(t2_path, low_ang_thre_deg = np.degrees(0.0005))
##lu.save_log(t2_straight, './sdcdata/dt2/driving_log_s.csv')
#
#t2_tough = lu.log_thresholding(t2_path, high_ang_thre_deg = 15)
##lu.save_log(t2_tough, './sdcdata/dt2/driving_log_t.csv')
#angles = lu.log_to_angles(t2_tough)
#print ('Track 2 tough ==================')
#print_stat(angles)
#
#
#t2_balansed = lu.balance_log(t2_path, undersampling_stds=20)
#angles = lu.log_to_angles(t2_balansed)
#print ('Track 2 balanced ==================')
#print_stat(angles)
#plot_hist(angles, 'Track 2 balanced')
##lu.save_log(t2_balansed, './sdcdata/dt2/driving_log_b.csv')

log = lu.readlog(log_path = t1_path,	img_path = './sdcdata/dt1/IMG/')	
#log = lu.readlog(log_path = t2_path,	img_path = './sdcdata/dt2/IMG/')	
fig1 = plt.figure(1,figsize=(9, 4))
cols = 3 
rows = 3
cellscnt = rows

#log = lu.readlog(log_path = t1_path,	img_path = './sdcdata/dt1/IMG/')	
log = lu.readlog(log_path = t2_path,	img_path = './sdcdata/dt2/IMG/')	
fig1 = plt.figure(1,figsize=(9, 4))
cols = 3 
rows = 2
cellscnt = cols*rows

#def sidecam_examples(line):
#	ang = line[3]
#	side_angle = 0.17
#	side_rnd = 0.025
#	img = cv2.imread(line[0])
#	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#	imgl = cv2.imread(line[1])
#	imgl = cv2.cvtColor(imgl, cv2.COLOR_BGR2RGB)
#	angl = side_angle + np.random.uniform(-side_rnd, side_rnd, 1)[0]
#	imgr = cv2.imread(line[2])
#	imgr = cv2.cvtColor(imgr, cv2.COLOR_BGR2RGB)
#	angr = -1.0*side_angle + np.random.uniform(-side_rnd, side_rnd, 1)[0]
#	return [imgl,np.degrees(angl),img,np.degrees(ang),imgr,np.degrees(angr)]
#
#samples = [sidecam_examples(line) for line in log[::1000][:cellscnt]]
#
#gs = gridspec.GridSpec(rows, cols, hspace=0.4, wspace=0.1)
#ax = [plt.subplot(gs[i]) for i in range(cellscnt)]
#for i in range(rows):
#	imgl,angl,img,ang,imgr,angr = samples[i]
#	ax[3*i].set_title(str(angl))
#	ax[3*i].imshow(imgl)
#	ax[3*i].axis('off')
#	ax[3*i+1].set_title(str(ang))
#	ax[3*i+1].imshow(img)
#	ax[3*i+1].axis('off')
#	ax[3*i+2].set_title(str(angr))
#	ax[3*i+2].imshow(imgr)
#	ax[3*i+2].axis('off')
#plt.show()

samples = [lu.get_sample(line,keep_direct_threshold = 1.0) for line in log[::1000][:cellscnt]]
gs = gridspec.GridSpec(rows, cols, hspace=0.0, wspace=0.1)
ax = [plt.subplot(gs[i]) for i in range(cellscnt)]
for i in range(cellscnt):
	img,ang = samples[i]
	img,ang = lu.distort(img,ang)
	#img = lu.randomize_light(img)
	#img = lu.crop(img, ((70, 25), (0, 0)))
	ax[i].set_title(str(np.degrees(ang)))
	ax[i].imshow(img)
	ax[i].axis('off')
plt.show()


#from keras.utils import plot_model
#model = m.create_model(input_shape = (65,320,3)) 
#model.summary()
#plot_model(model, to_file='model.png')
#lu.filter_log(t2_path, 20, './sdcdata/dt2/driving_log_thre20.csv')
#lu.filter_log(t2_path, 22, './sdcdata/dt2/driving_log_thre22.csv')
#print("completed")