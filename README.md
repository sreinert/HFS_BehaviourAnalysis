# HFS_BehaviourAnalysis

This repo is a collection of analysis code for behavioural data for the head-fixed schema learning project. 

Design: Mice traverse a virtual linear corridor (panda3d) by running on a treadmill, read-out by rotary encoder. There are n visual landmarks, k of which are part of a target sequence (ABC...). Licking in an active landmark (piezo sensor) will deliver a drop of soy milk (solenoid valve). 

Data: per training session one **config.yaml** file with the settings and one **position_log.csv** containing position, licks, rewards,... 

Main functions:
Daily_summary: parses an individual session or training day for a specified mouse and date
cohort_1_summary: contains training stage info per mouse of cohort one and can produce a learning curve per stage concatinating all relevant sessions for a mouse
weight_analysis: pulls info from google sheet monitoring the restriction level of cohort 1 animals and relates that to engagement in task (limited so far)
