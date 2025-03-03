import ruamel.yaml
yaml = ruamel.yaml.YAML()
import numpy as np

#settings needed: mouse, date, session, stage, landmarks, number of repeats (default like 500 reps for full corridor), resulting length of the corridor, gain, speed-threshold
#stage 1 is a corridor without any landmarks but rewards given at a fixed distance run
#stage 2 is a corridor without any landmarks but rewards given lick-triggered after a fixed distance run
#stage 3 is a corridor with all landmarks in pseudorandom order that repeats the number of times specified, at the landmark a reward is given auto
#stage 4 is a corridor with all landmarks in pseudorandom order that repeats the number of times specified, at the landmark a reward is given lick-triggered
#stage 5 is a corridor with just two landmarks in A A B B configuration that repeats the number of times specified, at landmark A a reward is given lick-triggered (path4 in notion)
#alternatively stage 5 is a corridor with just two landmarks in either A B or B A arranged pseudorandomly that repeats the number of times specified, at landmark A a reward is given lick-triggered (path5 in notion)
#stage 6 is the testing corridor with all landmarks in fixed order, rewards are given in a A A B B A A B B fashion (4 rewards per corridor)
#alternatively stage 6 is the testing corridor with all landmarks in fixed order, rewards are given in a A B A B A B A B fashion (4 rewards per corridor)

#a mouse specific settings file needs to be loaded and integrated into the corridor design
mouse = 'TAA0000066'
mouse_file = mouse + '.yaml'
with open(str(mouse_file), 'r') as yaml_file:
    mouse_settings = yaml.load(yaml_file)


#mapping of visual landmarks to odours
landmarks = {'0':'logs.png', '1':'grating1.jpg', '2':'tiles.png', '3':'big_light_rectangles.png', '4':'grass.png', '5':'big_dark_rectangles.png', '6':'leaves.png', '7':'waves.png', '8':'bark.png', '9':'big_light_circles.png'}

#define basic unit of repeat structure - stage dependent
stage = mouse_settings['stage']
setup = mouse_settings['setup']
logger_name = mouse_settings['basepath'] +  mouse_settings['date']
if setup == 1:
    monitor1_width = 1366
    daq_channel_3V1 = 'Dev1/port1/line2'
    daq_channel_3V2 = 'Dev1/port2/line3'
else:
    monitor1_width = 1920
    daq_channel_3V1 = 'Dev1/port1/line1'
    daq_channel_3V2 = 'Dev1/port1/line2'

if stage == 1:
    rulename = 'run-auto'
    repeat_unit = 'random_dots.png'
    collect_seq = [repeat_unit]
    for i in range(mouse_settings['num_reps']):
        collect_seq.append(repeat_unit)
    length = (mouse_settings['num_reps']-1)*9
    landmark_full = None
    goals = None
elif stage == 2:
    rulename = 'run-lick'
    repeat_unit = 'random_dots.png'
    collect_seq = [repeat_unit]
    for i in range(mouse_settings['num_reps']):
        collect_seq.append(repeat_unit)
    length = (mouse_settings['num_reps']-1)*9
    landmark_full = None
    goals = None
elif stage == 3 or stage == 4:
    rulename = 'olfactory_support'
    lm_ids = mouse_settings['landmark_ids']
    collect_ids = lm_ids
    collect_seq = ['random_dots.png', 'random_dots.png','random_dots.png']
    for i in range(mouse_settings['num_reps']):
        pseudo_random_order = np.random.permutation(lm_ids)
        collect_ids = np.append(collect_ids,pseudo_random_order,axis=0)
    
    for n in range(len(collect_ids)):
        collect_seq.append('random_dots.png')
        collect_seq.append('random_dots.png')
        collect_seq.append('random_dots.png')
        collect_seq.append(landmarks[str(collect_ids[n])])
        collect_seq.append(landmarks[str(collect_ids[n])])
        collect_seq.append('random_dots.png')
        collect_seq.append('random_dots.png')
        collect_seq.append('random_dots.png')
        collect_seq.append('random_dots.png')
        collect_seq.append('random_dots.png')
        collect_seq.append('random_dots.png')
        collect_seq.append('random_dots.png')

    # all_positions = np.arange(0, (len(collect_seq)-1)*9, 9)
    all_positions = np.arange(0, len(collect_seq)*9, 9)
    # landmark_startpositions = all_positions[5::10]+2
    landmark_endpositions = all_positions[7::12]+2
    print(landmark_endpositions)
    landmark_startpositions = landmark_endpositions-18
    landmark_full = np.zeros((len(landmark_startpositions),2))
    for p in range(len(landmark_startpositions)):
        landmark_full[p,:] = [landmark_startpositions[p], landmark_endpositions[p]]
    goals = landmark_full
    length = (len(collect_seq)-1)*9
    a = ruamel.yaml.comments.CommentedSeq([landmark_full.astype(int).tolist()])
    a.fa.set_flow_style()
    b = ruamel.yaml.comments.CommentedSeq([goals.astype(int).tolist()])
    b.fa.set_flow_style()
    ids = ruamel.yaml.comments.CommentedSeq(collect_ids.astype(int).tolist())
    ids.fa.set_flow_style()
    lms = ruamel.yaml.comments.CommentedSeq([collect_ids.astype(int).tolist()])
    lms.fa.set_flow_style()
    if stage == 3:
        assisted_goals = np.zeros((len(goals),2))
        for i in range(len(goals)):
            assisted_goals[i,0] = goals.astype(int)[i,0]+2
            assisted_goals[i,1] = goals.astype(int)[i,0]+3
        assisted_goals = ruamel.yaml.comments.CommentedSeq(assisted_goals.astype(int).tolist())
        assisted_goals.fa.set_flow_style()
    elif stage == 4:
        assisted_goals = None

elif stage == 5:
    rulename = 'olfactory_support'
    lm_ids = mouse_settings['landmark_ids']
    collect_ids = []
    goal_id = mouse_settings['goal_ids']
    lm_sequence = []
    bin_collect_ids = np.empty(shape=[0, 2])
    collect_seq = ['random_dots.png', 'random_dots.png','random_dots.png']
    for i in range(mouse_settings['num_reps']):
        for i in range(len(lm_ids)):
            collect_seq.append(landmarks[str(lm_ids[i])])
            collect_seq.append(landmarks[str(lm_ids[i])])
            collect_seq.append('random_dots.png')
            collect_seq.append(landmarks[str(lm_ids[i])])
            collect_seq.append(landmarks[str(lm_ids[i])])
            collect_seq.append('random_dots.png')
            lm_sequence = np.append(lm_sequence,[lm_ids[i], lm_ids[i]],axis=0)
        collect_ids = np.append(collect_ids,lm_sequence[-4:]==goal_id,axis=0)
        

    all_positions = np.arange(0, (len(collect_seq)-1)*9, 9)
    landmark_startpositions = all_positions[2::3]+2
    landmark_endpositions = landmark_startpositions+16
    landmark_full = np.zeros((len(landmark_startpositions),2))
    for p in range(len(landmark_startpositions)):
        landmark_full[p,:] = [landmark_startpositions[p], landmark_endpositions[p]]
    for i, e in enumerate(collect_ids):
        if e > 0:
            bin_collect_ids= np.append(bin_collect_ids,[landmark_full[i]],axis=0)
    goals = bin_collect_ids
    length = (len(collect_seq)-1)*9
    a = ruamel.yaml.comments.CommentedSeq([landmark_full.astype(int).tolist()])
    a.fa.set_flow_style()
    b = ruamel.yaml.comments.CommentedSeq([goals.astype(int).tolist()])
    b.fa.set_flow_style()
    ids = ruamel.yaml.comments.CommentedSeq(collect_ids.astype(int).tolist())
    ids.fa.set_flow_style()
    lms = ruamel.yaml.comments.CommentedSeq([lm_sequence.astype(int).tolist()])
    lms.fa.set_flow_style()
    assisted_goals = None
elif stage == 52:
    rulename = 'olfactory_support'
    lm_ids = mouse_settings['landmark_ids']
    goal_id = mouse_settings['goal_ids']
    collect_ids = []
    lm_sequence = []
    bin_collect_ids = np.empty(shape=[0, 2])
    collect_seq = ['random_dots.png', 'random_dots.png','random_dots.png']
    for i in range(mouse_settings['num_reps']):
        pseudo_random_order = np.random.permutation(lm_ids)
        # collect_ids.append(pseudo_random_order)
        lm_sequence =  np.append(lm_sequence,pseudo_random_order,axis=0)
        collect_ids = np.append(collect_ids,pseudo_random_order==goal_id,axis=0)

    for n in range(len(collect_ids)):
        collect_seq.append(landmarks[str(lm_sequence.astype(int)[n])])
        collect_seq.append(landmarks[str(lm_sequence.astype(int)[n])])
        collect_seq.append('random_dots.png')

    all_positions = np.arange(0, (len(collect_seq)-1)*9, 9)
    landmark_startpositions = all_positions[2::3]+2
    landmark_endpositions = landmark_startpositions+16
    landmark_full = np.zeros((len(landmark_startpositions),2))
    for p in range(len(landmark_startpositions)):
        landmark_full[p,:] = [landmark_startpositions[p], landmark_endpositions[p]]
    for i, e in enumerate(collect_ids):
        if e > 0:
            bin_collect_ids= np.append(bin_collect_ids,[landmark_full[i]],axis=0)
    goals = bin_collect_ids
    length = (len(collect_seq)-1)*9
    a = ruamel.yaml.comments.CommentedSeq([landmark_full.astype(int).tolist()])
    a.fa.set_flow_style()
    b = ruamel.yaml.comments.CommentedSeq([goals.astype(int).tolist()])
    b.fa.set_flow_style()
    ids = ruamel.yaml.comments.CommentedSeq(collect_ids.astype(int).tolist())
    ids.fa.set_flow_style()
    lms = ruamel.yaml.comments.CommentedSeq([lm_sequence.astype(int).tolist()])
    lms.fa.set_flow_style()
    assisted_goals = None
elif stage == 6 or stage == 62:
    rulename = 'olfactory_support'
    lm_ids = mouse_settings['landmark_ids']
    goal_id = mouse_settings['goal_ids']
    lm_sequence = []
    collect_ids = []
    bin_collect_ids = np.empty(shape=[0, 2])
    collect_seq = ['random_dots.png', 'random_dots.png','random_dots.png']
    for i in range(mouse_settings['num_reps']):
        lm_sequence = np.append(lm_sequence,lm_ids,axis=0)
        
        for i in range(len(lm_ids)):
            collect_seq.append(landmarks[str(lm_ids[i])])
            collect_seq.append(landmarks[str(lm_ids[i])])
            collect_seq.append('random_dots.png')

            if lm_ids[i] in goal_id:
                collect_ids = np.append(collect_ids,[goal_id.index(lm_ids[i])+1],axis=0)
            else:
                collect_ids = np.append(collect_ids,[0],axis=0) 

    all_positions = np.arange(0, (len(collect_seq)-1)*9, 9)
    landmark_startpositions = all_positions[2::3]+2
    landmark_endpositions = landmark_startpositions+16
    landmark_full = np.zeros((len(landmark_startpositions),2))
    for p in range(len(landmark_startpositions)):
        landmark_full[p,:] = [landmark_startpositions[p], landmark_endpositions[p]]
    for i, e in enumerate(collect_ids):
        if e > 0:
            bin_collect_ids= np.append(bin_collect_ids,[landmark_full[i]],axis=0)
    goals = bin_collect_ids
    length = (len(collect_seq)-1)*9
    a = ruamel.yaml.comments.CommentedSeq([landmark_full.astype(int).tolist()])
    a.fa.set_flow_style()
    b = ruamel.yaml.comments.CommentedSeq([goals.astype(int).tolist()])
    b.fa.set_flow_style()
    ids = ruamel.yaml.comments.CommentedSeq([collect_ids.astype(int).tolist()])
    ids.fa.set_flow_style()
    lms = ruamel.yaml.comments.CommentedSeq([lm_sequence.astype(int).tolist()])
    lms.fa.set_flow_style()
    assisted_goals = None
    

size = ruamel.yaml.comments.CommentedSeq([0.2, 0.2])
size.fa.set_flow_style()
position = ruamel.yaml.comments.CommentedSeq([0.9, 0.9])
position.fa.set_flow_style()
modd1_odours = ruamel.yaml.comments.CommentedSeq([1, 3, 5, 7, 9])
modd1_odours.fa.set_flow_style()
speed = {'chan': 'Dev1/ctr0', 'diameter': 0.197, 'pulses_per_rev': 1000, 'error_value': 4000000000, 'threshold': 0.001}
speed = ruamel.yaml.comments.CommentedMap(speed)
speed.fa.set_flow_style()
spout = {'chan': 'Dev1/ai10', 'min_value': 0, 'max_value': 10, 'threshold': 1}
spout = ruamel.yaml.comments.CommentedMap(spout)
spout.fa.set_flow_style()



#the fixed settings of corridor design can be specified here (collected in a big dictionary)
if stage == 1 or stage == 2:
    settings = {'texture_path': 'examples/textures/',
                'base_tunnel': {'speed_gain':mouse_settings['gain'],'eye_fov':{'fov':100,'fov_shift':57}, 'wall_model':'walls.egg', 'wall_length':4.5,'wall_spacing':9},
             'card':{'size':size,'position':position}, 
             'monitor':{'dual_monitor':True,'monitor1':{'width':monitor1_width},'monitor2':{'width':1920,'height':1080},'monitor3':{'width':1920,'height':1080}},
             'sequence_task':{'rulename':rulename,'protocol':'olfactory_support_l1'},
             'daqChannel':{'valve1':'Dev2/port0/line1',
                           'spout1':spout},
             'inputs':{'speed':speed},
             'logger':{'foldername':logger_name},
             'outputs':{},
             'flip_tunnel':{'sleep_time':0,
                            'stimulus_onset':12,
                            'neutral_texture':'grey.png',
                            'io_module':'nidaq',
                            'length':length,
                            'margin_start':9,
                            'reward_distance':20,
                            'reward_length':{'manual':0.1,'assist':0.15, 'correct':0.15,'wrong':0.15},
                            'reward_prob':1,
                            'manual_reward_with_space':True,
                            'sound_dir':'examples/sounds/',
                            'sounds':{'0':'6kHz_tone.ogg'},
                            'sound_at_A':False,
                            'reward_tone_length':1,
                            'modd1_odours':modd1_odours,
                            'odour_diffs':{'flush_same':1.5,'flush_other':0.75,'flush':1,'odour_overlap':0.05},
                            'continuous_corridor':True},
            'walls_sequence': collect_seq}
else:
    settings = {'texture_path': 'examples/textures/', 
                'base_tunnel': {'speed_gain':mouse_settings['gain'],'eye_fov':{'fov':100,'fov_shift':57}, 'wall_model':'walls.egg', 'wall_length':4.5,'wall_spacing':9},
                'card':{'size':size,'position':position}, 
                'monitor':{'dual_monitor':True,'monitor1':{'width':monitor1_width},'monitor2':{'width':1920,'height':1080},'monitor3':{'width':1920,'height':1080}},
                'sequence_task':{'rulename':rulename,'protocol':'olfactory_support_l1'},
                'daqChannel':{'valve1':'Dev2/port0/line1',
                            'spout1':spout,
                            'odour0':'Dev1/port2/line4',
                            'odour1':'Dev1/port0/line4',
                            'odour2':'Dev1/port1/line7',
                            'odour3':'Dev1/port0/line7',
                            'odour4':'Dev1/port1/line6',
                            'odour5':'Dev1/port0/line2',
                            'odour6':'Dev1/port2/line5',
                            'odour7':'Dev1/port0/line3',
                            'odour8':'Dev1/port1/line4',
                            'odour9':'Dev1/port0/line5',
                            'mo1':'Dev1/port0/line0',
                            'mo2':'Dev1/port2/line6',
                            'fiveV':'Dev1/port2/line3',
                            'finalV1':'Dev1/port1/line0',
                            'finalV2':'Dev1/port1/line3',
                            'threeV':daq_channel_3V1,
                            'threeV2':daq_channel_3V2},
                'inputs':{'speed':speed},
                'logger':{'foldername':logger_name},
                'outputs':{},
                'flip_tunnel':{'sleep_time':0,
                                'stimulus_onset':12,
                                'neutral_texture':'grey.png',
                                'io_module':'nidaq',
                                'length':length,
                                'margin_start':9,
                                'reward_distance':9,
                                'reward_length':{'manual':0.1,'assist':0.15, 'correct':0.15,'wrong':0.15},
                                'reward_prob':1,
                                'manual_reward_with_space':True,
                                'sound_dir':'examples/sounds/',
                                'sounds':{'0':'6kHz_tone.ogg'},
                                'sound_at_A': False,
                                'reward_tone_length':1,
                                'modd1_odours':modd1_odours,
                                'odour_diffs':{'flush_same':1.5,'flush_other':0.75,'flush':1,'odour_overlap':0.05},
                                'continuous_corridor':True,
                                'landmarks':[a],
                                'goals':[b],
                                'assisted_goals':[assisted_goals],
                                'landmarks_sequence': lms},
                                
                'goal_ids': ids,
                'walls_sequence': collect_seq}


save_name = mouse + '_' + str(mouse_settings['date']) + '_' + str(mouse_settings['session']) + '.yaml'
#the corridor design is saved as a yaml file that can be loaded by the corridor task
# yaml.default_flow_style = None
with open(str(save_name), 'w') as yaml_file:
    yaml.dump(settings, yaml_file)

