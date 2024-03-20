from pathlib import Path
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import sys
import yaml
from scipy.interpolate import make_interp_spline

#specify what data to load
mouse = 'SR_1136984'
date = '240318'
data_dir = '/Volumes/mrsic_flogel/public/projects/SaRe_20240219_hfs/training_data/v1/' + mouse + '/' + date
path = data_dir + '/position_log.csv'
data = pd.read_csv(path)
print(data.shape)

# data : table with Time, Position, Event, TotalRunDistance
position_idx = np.where(data['Position'] > -1)[0]
position = data['Position'][position_idx].values - 9
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
manual_reward_root_idx = np.where(data['Event'] == 'manually-rewarded')[0]
manual_reward_idx = data['Index'][manual_reward_root_idx].values
manual_reward_time = times[manual_reward_idx]
manual_reward_position = position[manual_reward_idx]


with open(str(data_dir + '/config.yaml'), 'r') as fd:
    options = yaml.load(fd, Loader=yaml.SafeLoader)    
goals = np.array(options['flip_tunnel']['goals']) - np.array(options['flip_tunnel']['margin_start'])
# goals = [[12, 21], [138, 147], [84, 93], [66, 75]]
# landmarks = np.array(options['flip_tunnel']['landmarks']) - 9
landmarks = [[12, 21], [30, 39], [48, 57], [66, 75], [84, 93], [102, 111], [120, 129], [138, 147], [156, 164], [173, 182]]

print(position.shape)
print(lick_idx.shape)
print(reward_idx.shape)
print(goals)

# fig, ax = plt.subplots(figsize=(30, 2), dpi=100)
# ax.autoscale(enable=True, axis='both')
# ax.plot(times, position, color='lightgray')

# for i in range(len(reward_time)-1):
#     rectangle = patches.Rectangle((reward_time[i], goals[i%3][0]), reward_delta_time[i], np.diff(goals[i%3])[0], edgecolor='white', facecolor='gray', alpha=0.8)
#     ax.add_patch(rectangle)
# if not lick_idx is None:
#     ax.scatter(times[lick_idx], position[lick_idx], marker='x', alpha=0.5, zorder=10)
# if not manual_reward_idx is None:
#     ax.scatter(times[manual_reward_idx], position[manual_reward_idx], marker='o', alpha=0.5, edgecolors='red', facecolors='none', zorder=20)
# # yticks = np.array([0, 30, 60, 90])
# yticks = np.array([0, 60, 120, 180])
# ax.set_yticks(yticks)
# ax.set_yticklabels(yticks)
# ax.set_ylabel('Position in the \n virtual corridor (0~180)')
# xticks = np.array([i for i in range(int(times.max() // 600 + 1))])
# ax.set_xticks(xticks*600)
# ax.set_xticklabels(xticks*10)
# ax.set_xlabel('Time from session start (min)')
# ax.set_title(str(reward_idx.shape[0]))
# plt.tight_layout()
# plt.show()

# I want a plot that per lap is essentially a histogram of licks per position. maybe a way to start is to actually have a histogram of licks per position
fig, ax = plt.subplots(figsize=(30, 2), dpi=100)
ax.hist(lick_position, bins=50)
ax.set_xlabel('Position in the \n virtual corridor (0~180)')
# Plotting goals as grey bars
for goal in goals:
    rectangle = patches.Rectangle((goal[0],-55), np.diff(goal)[0], 50, edgecolor='grey', facecolor='grey', alpha=0.5)
    ax.add_patch(rectangle)

plt.ylim(-100, 1000)
# plt.show()

#start by counting laps
total_dist = data['TotalRunDistance'][position_idx].values - 9
num_laps = np.ceil([total_dist.max()/position.max()])
#convert to int
num_laps = num_laps.astype(int)
print(num_laps)

# create a matrix of licks per lap per position bin
num_bins = 60
num_laps = num_laps[0]
licks_per_bin = np.zeros((num_laps, num_bins))
bin_edges = np.linspace(0, 180, num_bins+1)
for i in range(num_laps):
    lap_idx = np.where((total_dist >= i*position.max()) & (total_dist < (i+1)*position.max()))[0]
    #find overlapping values in lick_idx and lap_idx
    overlap = np.intersect1d(lick_idx, lap_idx)
    licks_per_lap = position[overlap]
    licks_per_bin[i], _ = np.histogram(licks_per_lap, bins=bin_edges)

#plot this matrix as an image
vmin=0
vmax=50
fig, ax = plt.subplots(figsize=(30, 5), dpi=100)
ax.imshow(licks_per_bin, aspect='auto', vmin=vmin, vmax=vmax)
ax.set_xticks([0, 20, 40, 60])
ax.set_xticklabels([0, 60, 120, 180])
ax.set_xlabel('Position in the \n virtual corridor (0~180)')
ax.set_ylabel('Lap number')
#show the goals as grey bars
for goal in goals:
    rectangle = patches.Rectangle((goal[0]/180*num_bins, -30), np.diff(goal)[0]/180*num_bins, 30, edgecolor='grey', facecolor='grey', alpha=0.5)
    ax.add_patch(rectangle)
#add a colorbar
cbar = plt.colorbar(ax.imshow(licks_per_bin, aspect='auto', vmin=vmin, vmax=vmax))
cbar.set_label('Number of licks')
plt.tight_layout()
# plt.show()

# for each traversal of the landmark zone, save if the animal licked or not
licks_per_landmark = np.zeros((len(landmarks), num_laps))
for i in range(num_laps):
    lap_idx = np.where((total_dist >= i*position.max()) & (total_dist < (i+1)*position.max()))[0]
    overlap = np.intersect1d(lick_idx, lap_idx)
    licks_per_lap = position[overlap]
    for j in range(len(landmarks)):
        goal_idx = np.where((licks_per_lap >= landmarks[j][0]) & (licks_per_lap <= landmarks[j][1]))[0]
        if len(goal_idx) > 0:
            licks_per_landmark[j, i] = 1

#plot this matrix as an image
fig, ax = plt.subplots(figsize=(30, 2), dpi=100)
ax.imshow(licks_per_landmark, aspect='auto')
ax.set_ylabel('Goal zone')
ax.set_xlabel('Lap number')

plt.tight_layout()
# plt.show()

# use the licks_per_landmark matrix to calculate the percentage of laps in which the animal licked in each goal zone
licks_per_landmark_percentage = np.sum(licks_per_landmark, axis=1)/num_laps*100
fig, ax = plt.subplots(figsize=(5, 2), dpi=100)
ax.bar(range(len(landmarks)), licks_per_landmark_percentage)
ax.set_xticks(range(len(landmarks)))
ax.set_xticklabels(range(1, len(landmarks)+1))
ax.set_xlabel('Goal zone')
ax.set_ylabel('Percentage of laps with a lick')
plt.tight_layout()
# plt.show()

# use the licks_per_landmark matrix to form a vector of the sequence of licked landmarks
#create an empty np vector that I can append to
lm_sequence = np.array([])
for i in range(num_laps):
    #find the indices of the landmarks that were licked in this lap and append to the sequence
    licked_landmarks = np.where(licks_per_landmark[:,i])[0]
    lm_sequence = np.append(lm_sequence, licked_landmarks)

# fig, ax = plt.subplots(figsize=(30, 2), dpi=100)
# plt.plot(lm_sequence)
# plt.show()


#calculate the transition matrix between landmarks 
transition_matrix = np.zeros((len(landmarks), len(landmarks)))

for i in range(len(landmarks)):
    lm_licks = np.where(lm_sequence == i)[0]
    print(lm_licks)
    following_licks = lm_sequence[lm_licks[:-1]+1]
    print(following_licks)
    #calculate the transition matrix
    for j in range(len(landmarks)):
        transition_matrix[i,j] = np.sum(following_licks == j)
    #normalize the transition matrix
    # transition_matrix = transition_matrix / np.sum(transition_matrix, axis=1)[:, np.newaxis]


# fig, ax = plt.subplots(figsize=(5, 5), dpi=100)
# cax = ax.matshow(transition_matrix, cmap='viridis')
# fig.colorbar(cax)
# ax.set_xlabel('Landmarks')
# ax.set_ylabel('Landmarks')
# plt.tight_layout()
# plt.show()

#calculate the ideal transition matrix
ideal_transition_matrix = np.zeros((len(landmarks), len(landmarks)))
# for sequence 1
# ideal_transition_matrix[1,7] = 1
# ideal_transition_matrix[7,4] = 1
# ideal_transition_matrix[4,3] = 1
# ideal_transition_matrix[3,1] = 1

# for sequence 2
ideal_transition_matrix[2,9] = 1
ideal_transition_matrix[9,5] = 1
ideal_transition_matrix[5,6] = 1
ideal_transition_matrix[6,2] = 1

# #plot the ideal transition matrix in a subplot next to the real transition matrix
# fig, ax = plt.subplots(figsize=(10, 5), dpi=100)
# cax = ax.matshow(transition_matrix, cmap='viridis')
# fig.colorbar(cax)
# ax.set_xlabel('Landmarks')
# ax.set_ylabel('Landmarks')
# ax.set_title('Real transition matrix')
# plt.tight_layout()

# ax2 = ax.twiny()
# cax2 = ax2.matshow(ideal_transition_matrix, cmap='viridis')
# fig.colorbar(cax2)
# ax2.set_xlabel('Landmarks')
# ax2.set_ylabel('Landmarks')
# ax2.set_title('Ideal transition matrix')
# plt.tight_layout()
# plt.show()

#plot the ideal transition matrix in a subplot next to the real transition matrix
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), dpi=100)

# Plot the real transition matrix
cax1 = ax1.matshow(transition_matrix, cmap='viridis')
fig.colorbar(cax1, ax=ax1)
ax1.set_xlabel('Landmarks')
ax1.set_ylabel('Landmarks')
ax1.set_title('Real transition matrix')

# Plot the ideal transition matrix
cax2 = ax2.matshow(ideal_transition_matrix, cmap='viridis')
fig.colorbar(cax2, ax=ax2)
ax2.set_xlabel('Landmarks')
ax2.set_ylabel('Landmarks')
ax2.set_title('Ideal transition matrix')

plt.tight_layout()
plt.show()
