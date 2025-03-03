from pathlib import Path
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import sys
import yaml
from scipy.interpolate import make_interp_spline
import wesanderson
from cycler import cycler
import palettes
import seaborn as sns
hfs_palette = palettes.met_brew('Hiroshige',n=123, brew_type="continuous")
rev_hfs = hfs_palette[::-1]
tm_palette = palettes.met_brew('Greek',n=123, brew_type="continuous")
color_scheme = wesanderson.film_palette('Grand Budapest Hotel')
custom_cycler = cycler(color=color_scheme)

# set path
mouse = 'TAA0000064'
date = '250228'
data_dir = '/Volumes/mrsic_flogel/public/projects/AtApSuKuSaRe_20250129_HFScohort2/' + mouse + '/TrainingData/' + date
    
# load position log data
path = data_dir + '/position_log.csv'
data = pd.read_csv(path)

# parse the individual variables from a table with Time, Position, Event, TotalRunDistance
position_idx = np.where(data['Position'] > -1)[0]
position = data['Position'][position_idx].values
times = data['Time'][position_idx].values
lick_root_idx = np.where(data['Event'] == 'challenged')[0]
lick_idx = data['Index'][lick_root_idx].values
lick_time = times[lick_idx]
lick_position = position[lick_idx]
reward_root_idx = np.where(data['Event'] == 'rewarded')[0]
reward_idx = data['Index'][reward_root_idx].values
reward_time = times[reward_idx]
reward_delta_time = np.diff(reward_time)
reward_position = position[reward_idx]
assistant_reward_root_idx = np.where(data['Event'] == 'assist-rewarded')[0]
assistant_reward_idx = data['Index'][assistant_reward_root_idx].values
assistant_reward_time = times[assistant_reward_idx]
assistant_reward_position = position[assistant_reward_idx]
manual_reward_root_idx = np.where(data['Event'] == 'manually-rewarded')[0]
manual_reward_idx = data['Index'][manual_reward_root_idx].values
manual_reward_time = times[manual_reward_idx]
manual_reward_position = position[manual_reward_idx]

with open(str(data_dir + '/config.yaml'), 'r') as fd:
    options = yaml.load(fd, Loader=yaml.SafeLoader)    
goals = np.array(options['flip_tunnel']['goals']) #- np.array(options['flip_tunnel']['margin_start'])
landmarks = np.array(options['flip_tunnel']['landmarks']) #- np.array(options['flip_tunnel']['margin_start'])
tunnel_length = options['flip_tunnel']['length']

total_dist = data['TotalRunDistance'][position_idx].values - np.array(options['flip_tunnel']['margin_start'])
num_laps = np.ceil([total_dist.max()/position.max()])
speed = np.diff(total_dist)/np.diff(times)
speed = np.append(speed, speed[-1])
thresholded_lick_idx = lick_idx[speed[lick_idx]<5]  

# plot the position log
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
ax.plot(times, position, color='black', linewidth=1)
ax.plot(lick_time, lick_position, 'o', color='red', markersize=3)
ax.plot(reward_time, reward_position, 'o', color='green', markersize=3)
ax.plot(assistant_reward_time, assistant_reward_position, 'o', color='blue', markersize=3)

# plot the speed
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
ax.plot(times, speed, color='black', linewidth=1)

#find the last landmark that was run through
last_landmark = np.where(landmarks[:,0] < position[-1])[0][-1]

#split the data into landmarks by selecting a window around each landmark start
num_bins = 30
window = np.linspace(-10,20,num_bins)
num_lms = last_landmark+1
seen_landmarks = landmarks[:num_lms]
licks_per_bin = np.zeros((num_lms, num_bins))
speed_per_bin = np.zeros((num_lms, num_bins))
for i,landmark in enumerate(seen_landmarks):
        for j in range(num_bins-1):
            start = landmark[0] + window[j]
            stop = landmark[0] + window[j+1]
            idx = np.where((position > start) & (position < stop))[0]
            licks_per_bin[i,j] = len(np.intersect1d(idx,lick_idx))
            speed_per_bin[i,j] = np.mean(speed[idx])

# print session summary
print('Session Summary:')
print('LMs completed:', num_lms)
print('Rewards given:', len(reward_idx))
print('Assistant rewards given:', len(assistant_reward_idx))
print('Licks made:', len(lick_idx))
print('Licks made below speed threshold:', len(thresholded_lick_idx))
print('Manual rewards given:', len(manual_reward_idx))

# plot the licks and speed per bin as images
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
ax.imshow(licks_per_bin[:,:-1], aspect='auto')
ax.set_xlabel('Bins')
ax.set_ylabel('Landmarks')
ax.set_title('Licks per bin')
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
ax.imshow(speed_per_bin[:,:-1], aspect='auto')
ax.set_xlabel('Bins')
ax.set_ylabel('Landmarks')
ax.set_title('Speed per bin')

av_speed_per_bin = np.nanmean(speed_per_bin, axis=0)
std_speed_per_bin = np.nanstd(speed_per_bin, axis=0)
sem_speed_per_bin = std_speed_per_bin/np.sqrt(num_lms)

av_speed_per_bin = av_speed_per_bin[:-1]
sem_speed_per_bin = sem_speed_per_bin[:-1]

av_licks_per_bin = np.nanmean(licks_per_bin, axis=0)
std_licks_per_bin = np.nanstd(licks_per_bin, axis=0)
sem_licks_per_bin = std_licks_per_bin/np.sqrt(num_lms)

av_licks_per_bin = av_licks_per_bin[:-1]
sem_licks_per_bin = sem_licks_per_bin[:-1]

fig, (ax1,ax2) = plt.subplots(2,1,figsize=(10, 6), dpi=100)
ax1.set_prop_cycle(custom_cycler)
ax1.plot(window[:-1],av_speed_per_bin)
ax1.fill_between(window[:-1], av_speed_per_bin-sem_speed_per_bin, av_speed_per_bin+sem_speed_per_bin, alpha=0.5)
ymax = np.max(av_speed_per_bin+sem_speed_per_bin)
rectangle = patches.Rectangle((0, 0), 18, ymax, edgecolor='grey', facecolor='grey', alpha=0.5)
ax1.add_patch(rectangle)
ax1.set_xlabel('Position in the \n virtual corridor')
ax1.set_ylabel('Running speed (cm/s)')
ax2.set_prop_cycle(custom_cycler)
ax2.plot(window[:-1],av_licks_per_bin)
ax2.fill_between(window[:-1], av_licks_per_bin-sem_licks_per_bin, av_licks_per_bin+sem_licks_per_bin, alpha=0.5)
ymax = np.max(av_licks_per_bin+sem_licks_per_bin)
rectangle = patches.Rectangle((0, 0), 18, ymax, edgecolor='grey', facecolor='grey', alpha=0.5)
ax2.add_patch(rectangle)
ax2.set_xlabel('Position in the \n virtual corridor')
ax2.set_ylabel('Licks')
plt.tight_layout()
plt.show()