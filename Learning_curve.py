from pathlib import Path
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import sys
import yaml
from scipy.interpolate import make_interp_spline
import Daily_summary as ds
from statistics import NormalDist
import wesanderson
import seaborn as sns
import palettes
hfs_palette = palettes.met_brew('Hiroshige',n=123, brew_type="continuous")
rev_hfs = hfs_palette[::-1]

def learning_curve(mouse,dates):
    #define performance variables
    hitrate = np.zeros(len(dates))
    farate = np.zeros(len(dates))
    d_prime = np.zeros(len(dates))
    lick_ratio = np.zeros(len(dates))
    lap_number = np.zeros(len(dates))

    cat_hitrate = []
    cat_skiprate = []
    cat_fa_rate = []

    #get the daily summary for dates
    for i,date in enumerate(dates):
        print("Summary for ",date)
        session = ds.analyze_session(mouse, date, False)
        hitrate[i] = session['hitrate']
        farate[i] = session['farate']
        if hitrate[i] == 0:
            hitrate[i] = 0.01
        elif hitrate[i] == 100:
            hitrate[i] = 99.99
        if farate[i] == 0:
            farate[i] = 0.01
        elif farate[i] == 100:
            farate[i] = 99.99
        d_prime[i] = NormalDist().inv_cdf(hitrate[i]/100)-NormalDist().inv_cdf(farate[i]/100)
        lick_ratio[i] = session['lick_ratio']
        lap_number[i] = session['num_laps']
        if i == 0:
            all_licks_per_bin = session['licks_per_bin']
        else:
            all_licks_per_bin = np.concatenate((all_licks_per_bin,session['licks_per_bin']),axis=0)

        cat_hitrate = np.concatenate((cat_hitrate,session['sliding_window_hitrate2']))
        cat_skiprate = np.concatenate((cat_skiprate,session['sliding_window_skiprate']))
        cat_fa_rate = np.concatenate((cat_fa_rate,session['sliding_window_farate']))

    #plot the learning curve (hit rate and false alarm rate in one plot, dprime in another, lick ratio in another, lap number in a third)
    fig, ax = plt.subplots(1,4,figsize=(20,4))
    ax[0].plot(hitrate, label='Hit rate')
    ax[0].plot(farate, label='False alarm rate')
    ax[0].set_ylim(0,100)
    ax[0].set_xlabel('Session')
    ax[0].set_ylabel('Percentage')
    ax[0].legend()
    ax[0].set_title('Hit/FA rate')
    ax[1].plot(d_prime)
    ax[1].set_ylim(0,5)
    ax[1].set_xlabel('Session')
    ax[1].set_ylabel('dprime')
    ax[1].set_title('Learning curve')
    ax[2].plot(lick_ratio)
    ax[2].set_xlabel('Session')
    ax[2].set_ylabel('Lick ratio')
    ax[2].set_title('Lick ratio (Goal vs Distractor)')
    ax[3].plot(lap_number)
    ax[3].set_xlabel('Session')
    ax[3].set_ylabel('Number of laps')
    ax[3].set_title('Number of laps')
    

    #plot all licks per bin as a heatmap and draw horizontal lines to separate the different days
    all_licks_per_bin = np.array(all_licks_per_bin)
    fig, ax = plt.subplots()
    sns.heatmap(all_licks_per_bin, ax=ax, cmap=rev_hfs)
    # ax.imshow(all_licks_per_bin, aspect='auto', vmin=0, vmax=20)
    for i in range(len(dates)):
        ax.axhline(np.sum(lap_number[:i]), color='white', linewidth=1)
    ax.set_title('Licks per bin mouse '+mouse)
    ax.set_xlabel('Position')
    ax.set_ylabel('Lap')

    #plot the category hit rate, skip rate and false alarm rate in one plot
    fig, ax = plt.subplots()
    ax.plot(cat_hitrate, label='Hit rate')
    ax.plot(cat_skiprate, label='Skip rate')
    ax.plot(cat_fa_rate, label='False alarm rate')
    ax.set_ylim(0,1)
    ax.set_xlabel('Laps')
    ax.set_ylabel('Percentage')
    ax.legend()
    ax.set_title('Concatenated hit/skip/false alarm rate')
    # draw vertical lines to separate the different days
    for i in range(len(dates)):
        ax.axvline(np.sum(lap_number[:i]), color='black', linewidth=1)
    plt.show()





#simple sequence
# learning_curve('SR_0000006', ['240828','240829', '240829_2', '240830', '240830_2', '240831', '240831_2', '240901', '240901_2', '240902', '240902_2'])
# learning_curve('SR_0000011', ['240828','240829', '240829_2', '240830', '240830_2', '240831', '240831_2'])

#discrimination
# learning_curve('SR_0000006', ['240822','240823', '240824', '240826'])
# learning_curve('SR_0000012', ['240822','240823', '240824', '240826'])

#sequence 2
# learning_curve('SR_0000010', ['240908', '240908_2', '240909', '240909_2', '240910', '240910_2', '240911', '240911_2'])
# learning_curve('SR_0000011', ['240901', '240901_2'])
learning_curve('SR_0000011', ['240901', '240901_2','240902', '240902_2','240903', '240903_2','240904', '240904_2'])

