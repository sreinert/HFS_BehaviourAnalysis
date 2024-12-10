import gspread
import json
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import wesanderson
from cycler import cycler
color_scheme = wesanderson.film_palette('Darjeeling Limited')
custom_cycler = cycler(color=color_scheme)

gc = gspread.oauth()

sh = gc.open('HFS cohort 1')
worksheet = sh.get_worksheet(0)
dates = worksheet.col_values(1)
dates = dates[1:]
weights1 = worksheet.col_values(9)
weights1 = weights1[1:]
weights1 = np.array(weights1)
for i in range(len(weights1)):
    if weights1[i] == '':
        weights1[i] = np.nan
    else:
        weights1[i] = float(weights1[i])
weights1 = weights1.astype(np.float64)
weights1[weights1 == 0] = np.nan
weights2 = worksheet.col_values(15)
weights2 = weights2[1:]
weights2 = np.array(weights2)
for i in range(len(weights2)):
    if weights2[i] == '':
        weights2[i] = np.nan
    else:
        weights2[i] = float(weights2[i])
weights2 = weights2.astype(np.float64)
weights2[weights2 == 0] = np.nan
weights3 = worksheet.col_values(21)
weights3 = weights3[1:]
weights3 = np.array(weights3)
for i in range(len(weights3)):
    if weights3[i] == '':
        weights3[i] = np.nan
    else:
        weights3[i] = float(weights3[i])
weights3 = weights3.astype(np.float64)
weights3[weights3 == 0] = np.nan
weights4 = worksheet.col_values(27)
weights4 = weights4[1:]
weights4 = np.array(weights4)
for i in range(len(weights4)):
    if weights4[i] == '':
        weights4[i] = np.nan
    else:
        weights4[i] = float(weights4[i])
weights4 = weights4.astype(np.float64)
weights4[weights4 == 0] = np.nan
weights5 = worksheet.col_values(33)
weights5 = weights5[1:]
weights5 = np.array(weights5)
for i in range(len(weights5)):
    if weights5[i] == '':
        weights5[i] = np.nan
    else:
        weights5[i] = float(weights5[i])
weights5 = weights5.astype(np.float64)
weights5[weights5 == 0] = np.nan
weights6 = worksheet.col_values(39)
weights6 = weights6[1:]
weights6 = np.array(weights6)
for i in range(len(weights6)):
    if weights6[i] == '':
        weights6[i] = np.nan
    else:
        weights6[i] = float(weights6[i])
weights6 = weights6.astype(np.float64)
weights6[weights6 == 0] = np.nan
weights7 = worksheet.col_values(45)
weights7 = weights7[1:]
weights7 = np.array(weights7)
for i in range(len(weights7)):
    if weights7[i] == '':
        weights7[i] = np.nan
    else:
        weights7[i] = float(weights7[i])
weights7 = weights7.astype(np.float64)
weights7[weights7 == 0] = np.nan

trial_num1 = worksheet.col_values(6)
trial_num1 = trial_num1[1:]
trial_num1 = np.array(trial_num1,dtype='<U3')
for i in range(len(trial_num1)):
    if trial_num1[i] == '':
        trial_num1[i] = np.nan
    else:
        trial_num1[i] = float(trial_num1[i])
trial_num1 = trial_num1.astype(np.float64)
trial_num2 = worksheet.col_values(12)
trial_num2 = trial_num2[1:]
trial_num2 = np.array(trial_num2,dtype='<U3')
for i in range(len(trial_num2)):
    if trial_num2[i] == '':
        trial_num2[i] = np.nan
    else:
        trial_num2[i] = float(trial_num2[i])
trial_num2 = trial_num2.astype(np.float64)
trial_num3 = worksheet.col_values(18)
trial_num3 = trial_num3[1:]
trial_num3 = np.array(trial_num3, dtype='<U3')
for i in range(len(trial_num3)):
    if trial_num3[i] == '':
        trial_num3[i] = np.nan
    else:
        trial_num3[i] = float(trial_num3[i])
trial_num3 = trial_num3.astype(np.float64)
trial_num4 = worksheet.col_values(24)
trial_num4 = trial_num4[1:]
trial_num4 = np.array(trial_num4,dtype='<U3')
for i in range(len(trial_num4)):
    if trial_num4[i] == '':
        trial_num4[i] = np.nan
    else:
        trial_num4[i] = float(trial_num4[i])
trial_num4 = trial_num4.astype(np.float64)
trial_num5 = worksheet.col_values(30)
trial_num5 = trial_num5[1:]
trial_num5 = np.array(trial_num5,dtype='<U3')
for i in range(len(trial_num5)):
    if trial_num5[i] == '':
        trial_num5[i] = np.nan
    else:
        trial_num5[i] = float(trial_num5[i])
trial_num5 = trial_num5.astype(np.float64)
trial_num6 = worksheet.col_values(36)
trial_num6 = trial_num6[1:]
trial_num6 = np.array(trial_num6,dtype='<U3')
for i in range(len(trial_num6)):
    if trial_num6[i] == '':
        trial_num6[i] = np.nan
    else:
        trial_num6[i] = float(trial_num6[i])
trial_num6 = trial_num6.astype(np.float64)
trial_num7 = worksheet.col_values(42)
trial_num7 = trial_num7[1:]
trial_num7 = np.array(trial_num7,dtype='<U3')
for i in range(len(trial_num7)):
    if trial_num7[i] == '':
        trial_num7[i] = np.nan
    else:
        trial_num7[i] = float(trial_num7[i])
trial_num7 = trial_num7.astype(np.float64)

rlevel1 = 1-np.array(weights1)
rlevel2 = 1-np.array(weights2)
rlevel3 = 1-np.array(weights3)
rlevel4 = 1-np.array(weights4)
rlevel5 = 1-np.array(weights5)
rlevel6 = 1-np.array(weights6)
rlevel7 = 1-np.array(weights7)


def exp_func(x,a,b,c):
    return a*np.exp(-b*x)+c

valid1x = np.where(~np.isnan(trial_num1) & ~np.isnan(weights1))
valid2x = np.where(~np.isnan(trial_num2) & ~np.isnan(weights2))
# valid3x = np.where(~np.isnan(trial_num3) & ~np.isnan(weights3))
valid4x = np.where(~np.isnan(trial_num4) & ~np.isnan(weights4))
valid5x = np.where(~np.isnan(trial_num5) & ~np.isnan(weights5))
valid6x = np.where(~np.isnan(trial_num6) & ~np.isnan(weights6))
valid7x = np.where(~np.isnan(trial_num7) & ~np.isnan(weights7))

popt1, pcov1 = curve_fit(exp_func, rlevel1[valid1x],trial_num1[valid1x])
try:
    popt2, pcov2 = curve_fit(exp_func, rlevel2[valid2x],trial_num2[valid2x])
except:
    print('Fitting Mouse 7 failed')
    popt2 = [0,0,0]

# popt3, pcov3 = curve_fit(exp_func, rlevel3[valid3x],trial_num3[valid3x])
popt4, pcov4 = curve_fit(exp_func, rlevel4[valid4x],trial_num4[valid4x])
popt5, pcov5 = curve_fit(exp_func, rlevel5[valid5x],trial_num5[valid5x])
popt6, pcov6 = curve_fit(exp_func, rlevel6[valid6x],trial_num6[valid6x])
popt7, pcov7 = curve_fit(exp_func, rlevel7[valid7x],trial_num7[valid7x])


print(dates)
print(weights1)

fig,ax = plt.subplots(1,1,figsize=(10,5))
ax.set_prop_cycle(custom_cycler)
ax.plot(dates[:len(weights1)],weights1,label='Mouse 6')
ax.plot(dates[:len(weights2)],weights2,label='Mouse 7')
# ax.plot(dates[:len(weights3)],weights3,label='Mouse 8')
# ax.plot(dates[:len(weights4)],weights4,label='Mouse 9')
ax.plot(dates[:len(weights5)],weights5,label='Mouse 10')
ax.plot(dates[:len(weights6)],weights6,label='Mouse 11')
ax.plot(dates[:len(weights7)],weights7,label='Mouse 12')
ax.set_xlabel('Date')
ax.set_ylabel('Weight (%)')
ax.legend()

level_range = range(-5, 20, 1)
level_range = np.array(level_range)/100

fig,ax = plt.subplots(1,1,figsize=(10,5))
ax.set_prop_cycle(custom_cycler)
ax.scatter(rlevel1[np.where(trial_num1)[0]]*100,trial_num1,label='Mouse 6')
ax.plot(level_range*100, exp_func(level_range, *popt1))
ax.scatter(rlevel2[np.where(trial_num2)[0]]*100,trial_num2,label='Mouse 7')
ax.plot(level_range*100, exp_func(level_range, *popt2))
# ax.scatter(trial_num3,weights3[np.where(trial_num3)[0]],label='Mouse 8')
# ax.scatter(trial_num4,weights4[np.where(trial_num4)[0]]*100,label='Mouse 9')
ax.scatter(rlevel5[np.where(trial_num5)[0]]*100,trial_num5,label='Mouse 10')
ax.plot(level_range*100, exp_func(level_range, *popt5))
ax.scatter(rlevel6[np.where(trial_num6)[0]]*100,trial_num6,label='Mouse 11')
ax.plot(level_range*100, exp_func(level_range, *popt6))
ax.scatter(rlevel7[np.where(trial_num7)[0]]*100,trial_num7,label='Mouse 12')
ax.plot(level_range*100, exp_func(level_range, *popt7))
ax.set_xlabel('Restriction level (%)')
ax.set_ylabel('Lap Number')
ax.legend()

plt.show()
