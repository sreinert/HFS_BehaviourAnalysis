# HFS_BehaviourAnalysis

This repo is a collection of analysis code for behavioural data for the head-fixed schema learning project. 

Design: Mice traverse a virtual linear corridor (panda3d) by running on a treadmill, read-out by rotary encoder. There are n visual landmarks, k of which are part of a target sequence (ABC...). Licking in an active landmark (piezo sensor) will deliver a drop of soy milk (solenoid valve). 

Data: per training session one **config.yaml** file with the settings and one **position_log.csv** containing position, licks, rewards,... 

