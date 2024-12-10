
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

#main function to parse position log file into behavioural parameters
def analyze_session(mouse, date, produce_plots):
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    
    # specify data directory
    data_dir = '/Volumes/mrsic_flogel/public/projects/SaRe_20240219_hfs/training_data/v2/' + mouse + '/' + date
    
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

    #load the config file to get the goals and landmarks
    with open(str(data_dir + '/config.yaml'), 'r') as fd:
        options = yaml.load(fd, Loader=yaml.SafeLoader)    
    goals = np.array(options['flip_tunnel']['goals']) #- np.array(options['flip_tunnel']['margin_start'])
    landmarks = np.array(options['flip_tunnel']['landmarks']) #- np.array(options['flip_tunnel']['margin_start'])
    tunnel_length = options['flip_tunnel']['length']
    
    #start by counting laps (repetitions of the corridor)
    total_dist = data['TotalRunDistance'][position_idx].values - np.array(options['flip_tunnel']['margin_start'])
    num_laps = np.ceil([total_dist.max()/position.max()])
    #convert to int
    num_laps = num_laps.astype(int)
    #create a variable that indexes the laps by finding flips first
    flip_ix = np.where(np.diff(position) < -50)[0]
    # a lap is between two flips
    lap_num = np.zeros(len(position))
    for i in range(len(flip_ix)-1):
        lap_num[flip_ix[i]:flip_ix[i+1]] = i+1
    lap_num[flip_ix[-1]:] = len(flip_ix)

    #calculate running speed and remove licks that occur above a certain speed threshold
    speed = np.diff(total_dist)/np.diff(times)
    speed = np.append(speed, speed[-1])
    thresholded_lick_idx = lick_idx[speed[lick_idx]<5]  

    #find indices of goals in the landmarks
    goal_idx = np.array([])
    for goal in goals:
        goal_idx = np.append(goal_idx, np.where(landmarks == goal)[0])
    goal_idx = goal_idx.astype(int)

    # create a matrix of licks per lap per position bin
    num_bins = 120
    num_laps = num_laps[0]
    licks_per_bin = np.zeros((num_laps, num_bins))
    bin_edges = np.linspace(0, position.max(), num_bins+1)
    #save lap_number as a variable with the same length as the position log
    for i in range(num_laps):
        lap_idx = np.where(lap_num==i)[0]
        #find overlapping values in lick_idx and lap_idx -> changed to thresholded_lick_idx
        overlap = np.intersect1d(thresholded_lick_idx, lap_idx)
        licks_per_lap = position[overlap]
        licks_per_bin[i], _ = np.histogram(licks_per_lap, bins=bin_edges)
    
    # for each traversal of the landmark zone, save if the animal licked or not
    licks_per_landmark = np.zeros((len(landmarks), num_laps))
    for i in range(num_laps):
        lap_idx = np.where(lap_num==i)[0]
        overlap = np.intersect1d(lick_idx, lap_idx)
        licks_per_lap = position[overlap]
        for j in range(len(landmarks)):
            lm_idx = np.where((licks_per_lap >= landmarks[j][0]) & (licks_per_lap <= landmarks[j][1]))[0]
            if len(lm_idx) > 0:
                licks_per_landmark[j, i] = 1


    lm_idx = np.zeros(len(position))
    for i in range(len(landmarks)):
        lm = landmarks[i]
        lm_entry = np.where((position > lm[0]) & (position < lm[1]))[0]
        lm_idx[lm_entry] = i+1

    licks_per_landmark_percentage = np.sum(licks_per_landmark, axis=1)/num_laps*100
    hitrate = np.mean(licks_per_landmark_percentage[goal_idx])
    farate = np.mean(licks_per_landmark_percentage[np.delete(np.arange(len(landmarks)), goal_idx)])

    #identify how many laps are needed to finish a sequence of goals (1 2 or 3 if the sequence is 4 goals of 10 landmarks)
    laps_needed = 1
    for i in range(len(goal_idx)-1):
        if goal_idx[i+1] - goal_idx[i] < 0:
            laps_needed += 1
    if goal_idx[0] - goal_idx[-1] > 0:
        laps_needed -= 1

    #create a matrix that indicates which landmarks were rewarded in each lap
    rewarded_lms = np.zeros((len(landmarks), num_laps))
    for i in range(num_laps):
        lap_idx = np.where(lap_num==i)[0]
        for j in range(len(landmarks)):
            lm = np.where(lm_idx == j+1)[0]
            target_ix = np.intersect1d(lap_idx,lm)
            target_rewards = np.intersect1d(target_ix,reward_idx)
            if len(target_rewards) > 0:
                rewarded_lms[j,i] = 1
            else:
                rewarded_lms[j,i] = 0

    #create a vector that indicates which goal is active in each lap
    active_goal = np.zeros((len(landmarks),num_laps))
    count = 0
    active_goal[0,0] = goal_idx[count]
    for i in range(num_laps):
        for j in range(len(landmarks)):
            active_goal[j,i] = goal_idx[count]
            if rewarded_lms[j,i]==1:
                count += 1
                if count == len(goal_idx):
                    count = 0


    #calculate a sliding window hitrate and farate
    window_size = 10
    hitrate_sw = np.zeros(num_laps)
    farate_sw = np.zeros(num_laps)
    skiprate_sw = np.zeros(num_laps)
    hitrate2_sw = np.zeros(num_laps)
    for i in range(num_laps):
        if i < window_size:
            hitrate_sw[i] = np.nan
            farate_sw[i] = np.nan
            skiprate_sw[i] = np.nan
            hitrate2_sw[i] = np.nan
        else:
            lap_range = range(i-window_size, i)
            licks_in_inactive_goal = np.sum(licks_per_landmark[:,lap_range]-rewarded_lms[:,lap_range], axis=1)/(window_size/laps_needed)
            licks_in_active_goal  = np.sum(rewarded_lms[:,lap_range], axis=1)/(window_size/laps_needed)
            licks_per_landmark_percentage = np.sum(licks_per_landmark[:,lap_range], axis=1)/window_size
            hitrate_sw[i] = np.mean(licks_per_landmark_percentage[goal_idx])
            farate_sw[i] = np.mean(licks_per_landmark_percentage[np.delete(np.arange(len(landmarks)), goal_idx)])
            skiprate_sw[i] = 1-np.mean(licks_in_inactive_goal[goal_idx])
            hitrate2_sw[i] = np.mean(licks_in_active_goal[goal_idx])

        

    # use the licks_per_landmark matrix to form a vector of the sequence of licked landmarks
    lm_sequence = np.array([])
    for i in range(num_laps):
        #find the indices of the landmarks that were licked in this lap and append to the sequence
        licked_landmarks = np.where(licks_per_landmark[:,i])[0]
        lm_sequence = np.append(lm_sequence, licked_landmarks)
    
    #calculate the transition matrix between landmarks 
    transition_matrix = np.zeros((len(landmarks), len(landmarks)))
    norm_tm = np.zeros((len(landmarks), len(landmarks)))

    for i in range(len(landmarks)):
        lm_licks = np.where(lm_sequence == i)[0]
        # print(lm_licks)
        following_licks = lm_sequence[lm_licks[:-1]+1]
        # print(following_licks)
        #calculate the transition matrix
        for j in range(len(landmarks)):
            transition_matrix[i,j] = np.sum(following_licks == j)
        #normalize the transition matrix
        norm_tm[i,:] = transition_matrix[i,:]/np.sum(transition_matrix[i,:])

    #create ideal transition matrix with only the goals
    ideal_transition_matrix = np.zeros((len(landmarks), len(landmarks)))
    for i in range(len(goal_idx)-1):
        ideal_transition_matrix[goal_idx[i], goal_idx[i+1]] = 1
        ideal_transition_matrix[goal_idx[i], goal_idx[i]] = 0
    ideal_transition_matrix[goal_idx[-1], goal_idx[0]] = 1

    #create another transition matrix where the animal licks the goals but in the wrong order
    sorted_goal_idx = np.sort(goal_idx)
    wrong_transition_matrix = np.zeros((len(landmarks), len(landmarks)))
    for i in range(len(sorted_goal_idx)-1):
        wrong_transition_matrix[sorted_goal_idx[i], sorted_goal_idx[i+1]] = 1
        wrong_transition_matrix[sorted_goal_idx[i], sorted_goal_idx[i]] = 0
    wrong_transition_matrix[sorted_goal_idx[-1], sorted_goal_idx[0]] = 1
    

    #calculate the average lick rate in each bin across all laps
    av_licks_per_bin = np.mean(licks_per_bin, axis=0)
    std_licks_per_bin = np.std(licks_per_bin, axis=0)
    sem_licks_per_bin = std_licks_per_bin/np.sqrt(num_laps)

    #calculate the average running speed in each bin and save in a matrix with the same dimensions as licks_per_bin (exclude the last lap)
    speed_per_bin = np.zeros((num_laps-1, num_bins))
    for i in range(num_laps-1):
        lap_idx = np.where(lap_num==i)[0]
        speed_per_lap = speed[lap_idx]
        bin_ix = np.digitize(position[lap_idx], bin_edges)
        for j in range(num_bins):
            speed_per_bin[i,j] = np.mean(speed_per_lap[bin_ix == j])
    av_speed_per_bin = np.nanmean(speed_per_bin, axis=0)
    std_speed_per_bin = np.nanstd(speed_per_bin, axis=0)
    sem_speed_per_bin = std_speed_per_bin/np.sqrt(num_laps)

    #calculate the average lick rate in each landmark zone from the av_licks_per_bin vector
    av_licks_per_landmark = np.zeros(len(landmarks))
    for i in range(len(landmarks)):
        lm_idx = np.where((bin_edges[:-1] >= landmarks[i][0]) & (bin_edges[:-1] <= landmarks[i][1]))[0]
        av_licks_per_landmark[i] = np.mean(av_licks_per_bin[lm_idx])
    
    #calculate the ratio of av_licks_per_landmark of goals to the av_licks_per_landmark of non-goals
    goal_av_licks = np.mean(av_licks_per_landmark[goal_idx])
    non_goal_av_licks = np.mean(np.delete(av_licks_per_landmark, goal_idx))
    lick_ratio = goal_av_licks/non_goal_av_licks


    # produce plots if specified
    if produce_plots:

        # plot 0 - average running speed per position
        fig, ax = plt.subplots(figsize=(10, 2), dpi=100)
        ax.set_prop_cycle(custom_cycler)
        ax.plot(av_speed_per_bin)
        ax.fill_between(range(num_bins), av_speed_per_bin-sem_speed_per_bin, av_speed_per_bin+sem_speed_per_bin, alpha=0.5)
        ax.set_xlabel('Position in the \n virtual corridor (0~180)')
        ax.set_ylabel('Running speed (cm/s)')
        ymax = np.nanmax(av_speed_per_bin+sem_speed_per_bin)
        for goal in goals:
            rectangle = patches.Rectangle((goal[0]/tunnel_length*num_bins, 0), np.diff(goal)[0]/tunnel_length*num_bins, ymax, edgecolor='grey', facecolor='grey', alpha=0.3)
            ax.add_patch(rectangle)
        plt.tight_layout()

        #plot the speed per bin as a heatmap with an inverted color scale
        fig, ax = plt.subplots(figsize=(10, 5), dpi=100)
        sns.heatmap(speed_per_bin, ax=ax, cmap=rev_hfs)
        ax.set_xlabel('Position in the \n virtual corridor (0~180)')
        ax.set_ylabel('Lap number')
        #draw vertical lines for the goals
        ax.vlines(x=goals/tunnel_length*num_bins, ymin=0, ymax=num_laps, color='grey', linestyle='--')
        plt.tight_layout()

        # alternaive plot 1 average lick rate per position
        fig, ax = plt.subplots(figsize=(10, 2), dpi=100)
        ax.set_prop_cycle(custom_cycler)
        ax.plot(av_licks_per_bin)
        ax.fill_between(range(num_bins), av_licks_per_bin-sem_licks_per_bin, av_licks_per_bin+sem_licks_per_bin, alpha=0.5)
        ax.set_xlabel('Position in the \n virtual corridor (0~180)')
        ax.set_ylabel('Lick rate')
        ymax = np.max(av_licks_per_bin+sem_licks_per_bin)
        for goal in goals:
            rectangle = patches.Rectangle((goal[0]/tunnel_length*num_bins, 0), np.diff(goal)[0]/tunnel_length*num_bins, ymax, edgecolor='grey', facecolor='grey', alpha=0.3)
            ax.add_patch(rectangle)
        plt.tight_layout()
        
        # plot 2 - matrix of licks per lap per position bin
        fig, ax = plt.subplots(figsize=(10, 5), dpi=100)
        sns.heatmap(licks_per_bin, ax=ax, cmap=rev_hfs)
        # ax.imshow(licks_per_bin, aspect='auto', vmin=0, vmax=20)
        ax.set_xlabel('Position in the \n virtual corridor (0~180)')
        ax.set_ylabel('Lap number')
        for goal in goals:
            rectangle = patches.Rectangle((goal[0]/tunnel_length*num_bins, -30), np.diff(goal)[0]/tunnel_length*num_bins, 30, edgecolor='grey', facecolor='grey', alpha=0.5)
            ax.add_patch(rectangle)
        plt.tight_layout()
        
        # plot 3 - matrix of licks per lap per landmark
        fig, ax = plt.subplots(figsize=(5, 2), dpi=100)
        sns.heatmap(licks_per_landmark, ax=ax, cmap=tm_palette)
        # ax.imshow(licks_per_landmark, aspect='auto')
        ax.set_yticks(range(len(landmarks)))
        ax.set_yticklabels(range(0, len(landmarks)))
        ax.set_ylabel('Landmarks')
        ax.set_xlabel('Lap number')
        plt.tight_layout()
        
        # plot 4 - percentage of laps in which the animal licked in each goal zone
        fig, ax = plt.subplots(figsize=(5, 2), dpi=100)
        ax.set_prop_cycle(custom_cycler)
        ax.bar(range(len(landmarks)), licks_per_landmark_percentage)
        ax.set_xticks(range(len(landmarks)))
        ax.set_xticklabels(range(1, len(landmarks)+1))
        ax.set_xlabel('Landmarks')
        ax.set_ylabel('Percentage of laps with a lick')
        ax.set_title('Hitrate: ' + str(np.round(hitrate)) + '%' + ' | ' + 'False alarm rate: ' + str(np.round(farate)) + '%')
        plt.tight_layout()
        
        # plot 5 - transition matrix
        fig, (ax1, ax2,ax3) = plt.subplots(1, 3, figsize=(7.5, 2.5), dpi=100)
        sns.heatmap(transition_matrix, ax=ax1, cmap=tm_palette)
        ax1.set_xlabel('Landmarks')
        ax1.set_ylabel('Landmarks')
        ax1.set_title('Real transition matrix')
        # plot ideal transition matrix
        sns.heatmap(ideal_transition_matrix, ax=ax2, cmap=tm_palette)
        # cax2 = ax2.matshow(ideal_transition_matrix, cmap='viridis')
        ax2.set_xlabel('Landmarks')
        ax2.set_ylabel('Landmarks')
        ax2.set_title('Ideal transition matrix')
        # plot wrong transition matrix
        sns.heatmap(wrong_transition_matrix, ax=ax3, cmap=tm_palette)
        # cax3 = ax3.matshow(wrong_transition_matrix, cmap='viridis')
        ax3.set_xlabel('Landmarks')
        ax3.set_ylabel('Landmarks')
        ax3.set_title('Disc. transition matrix')
        plt.tight_layout()

        # plot 6 - sliding window hitrate and farate, using wesanderson color palette
        fig, ax = plt.subplots(figsize=(5, 2), dpi=100)
        ax.set_prop_cycle(custom_cycler)
        # ax.plot(hitrate_sw, label='Hit rate')
        ax.plot(farate_sw, label='False alarm rate')
        ax.plot(skiprate_sw, label='Skip rate')
        ax.plot(hitrate2_sw, label='Hit rate2')
        
        ax.set_ylim(0, 1)
        ax.set_xlabel('Lap number')
        ax.set_ylabel('Percentage')
        ax.legend()
        ax.set_title('Sliding window hit/FA rate')
        plt.tight_layout()
        plt.show()
    
    
    #store key variables in a dictionary
    session_summary = {'mouse': mouse,
                       'date': date,
                       'tunnel_length': tunnel_length,
                       'num_laps': num_laps, 
                       'hitrate': hitrate, 
                       'farate': farate, 
                       'num_rewards': len(reward_idx), 
                       'num_assist_rewards': len(assistant_reward_idx), 
                       'num_licks': len(lick_idx), 
                       'num_thresholded_licks': len(thresholded_lick_idx), 
                       'num_manual_rewards': len(manual_reward_idx), 
                       'goals': goals,
                       'landmarks': landmarks,
                       'licks_per_bin': licks_per_bin,
                       'licks_per_landmark': licks_per_landmark,
                       'licks_per_landmark_percentage': licks_per_landmark_percentage,
                       'transition_matrix': transition_matrix,
                       'ideal_transition_matrix': ideal_transition_matrix,
                       'lick_ratio': lick_ratio,
                       'sliding_window_hitrate': hitrate_sw,
                       'sliding_window_farate': farate_sw,
                       'sliding_window_skiprate': skiprate_sw,
                       'sliding_window_hitrate2': hitrate2_sw,
                        }

    
    # print session summary
    print('Session Summary:')
    print('Laps completed:', num_laps)
    print('Lick ratio:', np.round(lick_ratio, 2))
    print('Hitrate:', np.round(hitrate))
    print('False alarm rate:', np.round(farate))
    print('Rewards given:', len(reward_idx))
    print('Assistant rewards given:', len(assistant_reward_idx))
    print('Licks made:', len(lick_idx))
    print('Licks made below speed threshold:', len(thresholded_lick_idx))
    print('Manual rewards given:', len(manual_reward_idx))
    print(goals)
    print(options['flip_tunnel']['modd1_odours'])

    return session_summary

# run this on example sessions (typically 2 sessions per day)
session_1 = analyze_session('SR_0000012', '241119', False)
session_2 = analyze_session('SR_0000012', '241119_2', False)

# print a summary of the concatenated sessions
print('Concatenated Session Summary:')
print('Laps completed:', session_1['num_laps'] + session_2['num_laps'])
print('Hitrate:', np.round((session_1['hitrate'] + session_2['hitrate'])/2))
print('False alarm rate:', np.round((session_1['farate'] + session_2['farate'])/2))
print('Lick ratio:', np.round((session_1['lick_ratio'] + session_2['lick_ratio'])/2))
print('Rewards given:', session_1['num_rewards'] + session_2['num_rewards'])
print('Assistant rewards given:', session_1['num_assist_rewards'] + session_2['num_assist_rewards'])
print('Manual rewards given:', session_1['num_manual_rewards'] + session_2['num_manual_rewards'])

