import numpy as np
import torch
import time
import random
from matplotlib.ticker import ScalarFormatter
from environment_simple import Env_cellular as env_simple
from environment_egc import Env_cellular as env_egc
from environment_mrc import Env_cellular as env_mrc
from environment_sc import Env_cellular as env_sc
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
from PER_DDPG import DDPG as PER_DDPG
import PER_buffer
import utils
from td3_torch import Agent as TD3_AGENT
from ddpg_torch import Agent as DDPG_AGENT
from PPO import PPO
from CER_Buffer import CompositeReplayBuffer
from CER_DDPG import CER_DDPG
import warnings;
warnings.filterwarnings('ignore');
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
##################### System Parameters ####################
Emax = 0.2
K = 2
B = 3
T = 1
eta = 0.7
Pn = 2  # grant-based user's power
Pmax = 0.2
w_csk = 0.000001
w_d = 1.5 * (10 ** -5)
w_egc = 1 * (10 ** -6)
w_mrc = 2 * (10 ** -6)
##################### Hyper Parameters #####################
MAX_EPISODES = 100
MAX_EP_STEPS = 100
LR_A = 0.0002    # learning rate for actor
LR_C = 0.0004    # learning rate for critic
GAMMA = 0.9     # reward discount
TAU = 0.01      # soft replacement
MEMORY_CAPACITY = 10000
BATCH_SIZE = 32
s_dim = 3# dimsion of states
a_dim = 1# dimension of action
a_bound = 1 #bound of action
state_am = 1000
verbose = True
###################### Network Nodes Deployment  #######################
location_vector = np.array([[0, 1],[0,1000]]) #locations of GB users
location_GF = np.array([[1,1]])# np.ones((1, 2))
############################### Fadings ################################
#### fading for GF user (EHD) ######
h0x1 = np.random.randn(1, 1)
h0x2 = np.random.randn(1, 1)
#fading_0 = h0x1[0,0] ** 2 + h0x2[0,0] ** 2
fading_0 = 1
##### fading for GB users (PDs) ###
#hnx = np.array([[0.78954, 0.89234, 0.96456, 1.12345, 1.24553, 1.38754],[0.75324, 0.86986, 0.99322, 1.11114, 1.22532, 1.35143]])
hnx = np.array([[0.78345, 0.96553, 1.2854],[0.80114, 0.93532, 1.3043]])
# Gains between PD and Base Station
hnPD = np.array([0.89643, 0.95451])
fading_n_ = [0,0]
##################### Gains Passing Approach ##########################
diversity_mode = int(input(print("Enter the diversity technique to harvest energy (0:Simple, 1:EGC, 2:MRC, 3:SC): ")))
if diversity_mode==1:        # egc1
        print("Harvesting mode: EGC")
        for i in range (K):
                for j in range(B):
                        fading_n_[i] = fading_n_[i] + (hnx[i,j])
                fading_n_[i] = fading_n_[i] ** 2
        print("fading_n_: ", fading_n_)
        fading_n = np.vstack((fading_n_,hnPD))
        fading_n = np.matrix.transpose(fading_n)
        print("FADING_N: ", fading_n)
elif diversity_mode==2:
        print("Harvesting mode: MRC")
        for i in range (K):
                for j in range(B):
                        fading_n_[i] = fading_n_[i] + (hnx[i,j] ** 2)
        print("fading_n_: ", fading_n_)
        fading_n = np.vstack((fading_n_,hnPD))
        fading_n = np.matrix.transpose(fading_n)
        print("FADING_N: ", fading_n)
elif diversity_mode==3:
        print("Harvesting mode: SC")
        for i in range (K):
                fading_n_[i] = max(hnx[i,0] ** 2, hnx[i,1] ** 2, hnx[i,2] ** 2)
        print("fading_n_: ", fading_n_)
        fading_n = np.vstack((fading_n_,hnPD))
        fading_n = np.matrix.transpose(fading_n)
        print("FADING_N: ", fading_n)
else:
        print("Harvesting mode: Simple") # 1 antenna
        for i in range(K):
              fading_n_[i] = random.choices(hnx[i], k=1)
        fading_N = [item for sublist in fading_n_ for item in sublist]
        print("FADING_N_SIMPLE: ", fading_N)
        fading_n = np.vstack((fading_N,hnPD))
        fading_n = np.matrix.transpose(fading_n)


if diversity_mode==1:
        myenv = env_egc(MAX_EP_STEPS, s_dim, location_vector, location_GF, Emax, K, B, T, eta, Pn, Pmax, w_d, w_egc, w_csk, fading_n, fading_0)
elif diversity_mode==2:
        myenv = env_mrc(MAX_EP_STEPS, s_dim, location_vector, location_GF, Emax, K, B, T, eta, Pn, Pmax, w_d, w_mrc, w_csk, fading_n, fading_0)
elif diversity_mode==3:
        myenv = env_sc(MAX_EP_STEPS, s_dim, location_vector, location_GF, Emax, K, B, T, eta, Pn, Pmax, w_d, w_csk, fading_n, fading_0)
else:
        myenv = env_simple(MAX_EP_STEPS, s_dim, location_vector, location_GF, Emax, K, T, eta, Pn, Pmax, w_csk, fading_n, fading_0)
#Set random seed number
seed = 0
#-----Set seeds-------
torch.manual_seed(seed)
np.random.seed(seed)
#--------------------
#---------------------------------Initializing DDPG POLICY--------------------------------------------------------------
ddpg_agent=DDPG_AGENT(LR_A,LR_C,s_dim,TAU,a_bound,GAMMA,n_actions=1,max_size=MEMORY_CAPACITY,batch_size=BATCH_SIZE)
#-----------------------------------------------------------------------------------------------------------------------
#---------------------------------Initializing PER-DDPG POLICY----------------------------------------------------------
prioritized_replay_alpha=0.6
prioritized_replay_beta0=0.4
prioritized_replay_beta_iters=None
prioritized_replay_eps=10000
policy_per_ddpg = PER_DDPG(s_dim, a_dim, a_bound)
replay_buffer = PER_buffer.PrioritizedReplayBuffer(MEMORY_CAPACITY, prioritized_replay_alpha)
if prioritized_replay_beta_iters is None:
    prioritized_replay_beta_iters = MAX_EP_STEPS*MAX_EPISODES
beta_schedule = utils.LinearSchedule(prioritized_replay_beta_iters, initial_p=prioritized_replay_beta0, final_p=1.0)
#-----------------------------------------------------------------------------------------------------------------------
#---------------------------------Initializing CER-DDPG POLICY----------------------------------------------------------
initial_alpha=0
final_alpha=0.5
alpha_schedule = utils.LinearSchedule(MAX_EPISODES*MAX_EP_STEPS, initial_p=initial_alpha, final_p=final_alpha)
eeta=0.5
replay_buffer_cer=CompositeReplayBuffer(MEMORY_CAPACITY)
cer_agent=CER_DDPG(s_dim, a_dim, a_bound,eeta)
#-----------------------------------------------------------------------------------------------------------------------
#---------------------------------Initializing TD3 POLICY---------------------------------------------------------------
td3_agent=TD3_AGENT(LR_A,LR_C,s_dim,TAU,a_bound,GAMMA,update_actor_interval=2,n_actions=1,max_size=MEMORY_CAPACITY,batch_size=BATCH_SIZE)
#-----------------------------------------------------------------------------------------------------------------------
#---------------------------------initialize PPO POLICY-----------------------------------------------------------------
has_continuous_action_space = True  # continuous action space; else discret
action_std = 0.6                    # starting std for action distribution (Multivariate Normal)
action_std_decay_rate = 0.05        # linearly decay action_std (action_std = action_std - action_std_decay_rate)
min_action_std = 0.1                # minimum action_std (stop decay after action_std <= min_action_std)
action_std_decay_freq = int(2.5e6)  # action_std decay frequency (in num timesteps)
update_timestep = MAX_EP_STEPS * 3     # update policy every n timesteps
K_epochs = 40            # update policy for K epochs in one PPO update
eps_clip = 0.2          # clip parameter for PPO
ppo_agent = PPO(s_dim, a_dim, LR_A,LR_C, GAMMA, K_epochs, eps_clip, has_continuous_action_space, action_std)
#-----------------------------------------------------------------------------------------------------------------------
var = 1  # control exploration
total_time = 1
t_0 = time.time()

ep_rewardall_ddpg = []
ep_rewardall_per = []
ep_rewardall_cer = []
ep_rewardall_td3 = []
ep_rewardall_ppo = []
ep_rewardall_greedy = []
ep_rewardall_random = []

eh_rewardall_ddpg = []
eh_rewardall_per = []
eh_rewardall_cer = []
eh_rewardall_td3 = []
eh_rewardall_ppo = []
eh_rewardall_greedy = []
eh_rewardall_random = []

alphan_all_ddpg = []
alphan_all_per = []
alphan_all_cer = []
alphan_all_td3 = []
alphan_all_ppo = []
alphan_all_greedy = []
alphan_all_random = []

for i in range(MAX_EPISODES):
    batter_ini = myenv.reset()
    s = myenv.channel_sequence[i % myenv.K, :].tolist()  # the current GB user, 2 element [GB-GF, GB-BS]
    s.append(batter_ini)
    s = np.reshape(s, (1, s_dim))
    s = s * state_am  # amplify the state
    s_ddpg = s
    s_td3 = s
    s_ppo = s
    s_per = s
    s_cer = s
    s_greedy = s
    s_random = s

    ep_reward_ddpg = 0
    ep_reward_per = 0
    ep_reward_cer = 0
    ep_reward_td3 = 0
    ep_reward_ppo = 0
    ep_reward_random = 0
    ep_reward_greedy = 0

    eh_reward_ddpg = 0
    eh_reward_per = 0
    eh_reward_cer = 0
    eh_reward_td3 = 0
    eh_reward_ppo = 0
    eh_reward_random = 0
    eh_reward_greedy = 0

    alphan_ddpg = 0
    alphan_per = 0
    alphan_cer = 0
    alphan_td3 = 0
    alphan_ppo = 0
    alphan_greedy = 0
    alphan_random = 0

    s_traj_ddpg = []
    s_traj_per = []
    s_traj_cer = []
    s_traj_td3 = []
    s_traj_ppo = []
    s_traj_random = []
    s_traj_greedy = []

    for j in range(MAX_EP_STEPS):
        ######################## DDPG ########################
        a_ddpg = ddpg_agent.choose_action(s_ddpg)
        a_ddpg = np.clip(np.random.normal(a_ddpg, var), 0, 1)  # add randomness to action selection for exploration
        r_ddpg, s_ddpg_, EHD, done, alphan_d = myenv.step(a_ddpg, s_ddpg / state_am, j)
        alphan_ddpg += alphan_d
        s_ddpg_ = s_ddpg_ * state_am
        s_traj_ddpg.append(s_ddpg_)
        ddpg_agent.remember(s_ddpg[0], a_ddpg, r_ddpg, s_ddpg_[0], 0)
        ep_reward_ddpg += int(*r_ddpg)
        eh_reward_ddpg += EHD
        ######################## PER-DDPG ########################
        a_per = policy_per_ddpg.get_action(np.array(s_per))
        a_per = np.clip(np.random.normal(a_per, var), 0, 1)
        r_per, s_per_, eh_per, done, alphan_p = myenv.step(a_per, s_per / state_am, j)
        alphan_per += alphan_p
        s_per_ = s_per_ * state_am
        s_traj_per.append(s_per_)
        ep_reward_per += int(*r_per)
        eh_reward_per += eh_per
        replay_buffer.add(s_per[0], a_per, r_per, s_per_[0], 0)
        ######################## CER-DDPG ########################
        a_cer = cer_agent.get_action(np.array(s_cer))
        a_cer = np.clip(np.random.normal(a_cer, var), 0, 1)
        r_cer, s_cer_, eh_cer, done, alphan_c = myenv.step(a_cer, s_cer / state_am, j)
        alphan_cer += alphan_c
        s_cer_ = s_cer_ * state_am
        s_traj_cer.append(s_cer_)
        ep_reward_cer += int(*r_cer)
        eh_reward_cer += eh_cer
        replay_buffer_cer.add(s_cer[0], a_cer, r_cer, s_cer_[0], 0)
        ######################## TD3 ########################
        a_td3 = td3_agent.choose_action(np.array(s_td3))
        # Apply exploration noise to action
        a_td3 = np.clip(np.random.normal(a_td3, var), 0, 1)
        r_td3, s_td3_, eh_td3, done, alphan_t = myenv.step(a_td3, s_td3 / state_am, j)
        alphan_td3 += alphan_t
        s_td3_ = s_td3_ * state_am
        s_traj_td3.append(s_td3_)
        ep_reward_td3 += int(*r_td3)
        eh_reward_td3 += eh_td3
        td3_agent.remember(s_td3[0], a_td3, r_td3, s_td3_[0], 0)
        ######################## PPO ########################
        a_ppo = ppo_agent.select_action(s_ppo)
        a_ppo = np.clip(np.random.normal(a_ppo, var), 0, 1)
        r_ppo, s_ppo_, eh_ppo, done, alphan_P = myenv.step(a_ppo, s_ppo / state_am, j)
        alphan_ppo += alphan_P
        s_ppo_ = s_ppo_ * state_am
        s_traj_ppo.append(s_ppo_)
        ep_reward_ppo += int(*r_ppo)
        eh_reward_ppo += eh_ppo
        ppo_agent.buffer.rewards.append(r_ppo)
        if j == MAX_EP_STEPS - 1:
            ppo_agent.buffer.is_terminals.append(1)
        else:
            ppo_agent.buffer.is_terminals.append(0)
        ######################## Greedy ########################
        r_greedy, s_next_greedy, EHG, done, alphan_G = myenv.step_greedy(s_greedy / state_am, j)
        alphan_greedy += alphan_G
        s_traj_greedy.append(s_next_greedy)
        s_greedy = s_next_greedy * state_am
        ep_reward_greedy += (r_greedy)
        eh_reward_greedy += EHG
        ######################## Random ########################
        r_random, s_next_random, EHR, done, alphan_R = myenv.step_random(s_random / state_am, j)
        alphan_random += alphan_R
        s_traj_random.append(s_next_random)
        s_random = s_next_random * state_am
        ep_reward_random += r_random
        eh_reward_random += EHR

        ########################### Network Trainings ######################
        if var > 0.1:
            var *= .9998  # decay the action randomness
        # ----------------DDPG ALGORITHM TRAINING-----------------------
        ddpg_agent.learn()
        # ----------------PER-DDPG ALGORITHM TRAINING-----------------------
        beta_value = 0
        beta_value = beta_schedule.value(total_time)
        policy_per_ddpg.train(replay_buffer, True, beta_value, prioritized_replay_eps, BATCH_SIZE, GAMMA, TAU)
        # ----------------CER-DDPG ALGORITHM TRAINING-----------------------
        alpha = 0
        alpha = alpha_schedule.value(total_time)
        cer_agent.train(replay_buffer_cer, True, beta_value, prioritized_replay_eps, alpha, BATCH_SIZE, GAMMA, TAU)
        # ----------------TD3 ALGORITHM TRAINING-----------------------
        td3_agent.learn()
        # ----------------PPO ALGORITHM TRAINING-----------------------
        if total_time % update_timestep == 0:
            ppo_agent.update()
        if has_continuous_action_space and total_time % action_std_decay_freq == 0:
            ppo_agent.decay_action_std(action_std_decay_rate, min_action_std)
        ########################### Update States ######################
        s_ddpg = s_ddpg_
        s_per = s_per_
        s_cer = s_cer_
        s_td3 = s_td3_
        s_ppo = s_ppo_
        total_time += 1

        if j == MAX_EP_STEPS-1:
            #print(f"Episode: {i}, reward is {ep_reward}, and Explore is {var}")
            print('Episode:', i, ' DDPG: %i' % int(ep_reward_ddpg),' PER: %i' % int(ep_reward_per),
                  'CER: %i' % int(ep_reward_cer), 'TD3: %i' % int(ep_reward_td3), 'PPO: %i' % int(ep_reward_ppo),
                   'Greedy: %i' % int(ep_reward_greedy), 'Random: %i' % int(ep_reward_random))
            break

    ep_reward_ddpg = np.reshape(ep_reward_ddpg/MAX_EP_STEPS, (1,))
    ep_rewardall_ddpg.append(ep_reward_ddpg)
    eh_reward_ddpg = np.reshape(eh_reward_ddpg / MAX_EP_STEPS, (1,))
    eh_rewardall_ddpg.append(eh_reward_ddpg)
    alphan_ddpg = np.reshape(alphan_ddpg/MAX_EP_STEPS, (1,))
    alphan_all_ddpg.append(alphan_ddpg)

    ep_reward_per = np.reshape(ep_reward_per / MAX_EP_STEPS, (1,))
    ep_rewardall_per.append(ep_reward_per)
    eh_reward_per = np.reshape(eh_reward_per / MAX_EP_STEPS, (1,))
    eh_rewardall_per.append(eh_reward_per)
    alphan_per = np.reshape(alphan_per/MAX_EP_STEPS, (1,))
    alphan_all_per.append(alphan_per)

    ep_reward_cer = np.reshape(ep_reward_cer / MAX_EP_STEPS, (1,))
    ep_rewardall_cer.append(ep_reward_cer)
    eh_reward_cer = np.reshape(eh_reward_cer / MAX_EP_STEPS, (1,))
    eh_rewardall_cer.append(eh_reward_cer)
    alphan_cer = np.reshape(alphan_cer/MAX_EP_STEPS, (1,))
    alphan_all_cer.append(alphan_cer)

    ep_reward_td3 = np.reshape(ep_reward_td3/MAX_EP_STEPS, (1,))
    ep_rewardall_td3.append(ep_reward_td3)
    eh_reward_td3 = np.reshape(eh_reward_td3 / MAX_EP_STEPS, (1,))
    eh_rewardall_td3.append(eh_reward_td3)
    alphan_td3 = np.reshape(alphan_td3/MAX_EP_STEPS, (1,))
    alphan_all_td3.append(alphan_td3)

    ep_reward_ppo = np.reshape(ep_reward_ppo/MAX_EP_STEPS, (1,))
    ep_rewardall_ppo.append(ep_reward_ppo)
    eh_reward_ppo = np.reshape(eh_reward_ppo / MAX_EP_STEPS, (1,))
    eh_rewardall_ppo.append(eh_reward_ppo)
    alphan_ppo = np.reshape(alphan_ppo/MAX_EP_STEPS, (1,))
    alphan_all_ppo.append(alphan_ppo)

    ep_reward_greedy = np.reshape(ep_reward_greedy/MAX_EP_STEPS, (1,))
    ep_rewardall_greedy.append(ep_reward_greedy)
    eh_reward_greedy = np.reshape(eh_reward_greedy / MAX_EP_STEPS, (1,))
    eh_rewardall_greedy.append(eh_reward_greedy)
    alphan_greedy = np.reshape(alphan_greedy/MAX_EP_STEPS, (1,))
    alphan_all_greedy.append(alphan_greedy)

    ep_reward_random = np.reshape(ep_reward_random/MAX_EP_STEPS, (1,))
    ep_rewardall_random.append(ep_reward_random)
    eh_reward_random = np.reshape(eh_reward_random / MAX_EP_STEPS, (1,))
    eh_rewardall_random.append(eh_reward_random)
    alphan_random = np.reshape(alphan_random/MAX_EP_STEPS, (1,))
    alphan_all_random.append(alphan_random)

    print("Episodic alpha_n: ",alphan_per)

######### CALCULATING AVERAGE REWARDS AND EH ##########
avg_alphan = [[sum(alphan_all_ddpg)/len(alphan_all_ddpg)],
                [sum(alphan_all_per)/len(alphan_all_per)],
                [sum(alphan_all_cer)/len(alphan_all_cer)],
                [sum(alphan_all_td3)/len(alphan_all_td3)],
                [sum(alphan_all_ppo)/len(alphan_all_ppo)],
                [sum(alphan_all_greedy)/len(alphan_all_greedy)],
                [sum(alphan_all_random)/len(alphan_all_random)]]


avg_rewards = [sum(ep_rewardall_ddpg)/len(ep_rewardall_ddpg),
                sum(ep_rewardall_per)/len(ep_rewardall_per),
                sum(ep_rewardall_cer)/len(ep_rewardall_cer),
                sum(ep_rewardall_td3)/len(ep_rewardall_td3),
                sum(ep_rewardall_ppo)/len(ep_rewardall_ppo),
                sum(ep_rewardall_greedy)/len(ep_rewardall_greedy),
                sum(ep_rewardall_random)/len(ep_rewardall_random)]

avg_harvested_energy = [sum(eh_rewardall_ddpg)/len(eh_rewardall_ddpg),
                        sum(eh_rewardall_per)/len(eh_rewardall_per),
                        sum(eh_rewardall_cer)/len(eh_rewardall_cer),
                        sum(eh_rewardall_td3)/len(eh_rewardall_td3),
                        sum(eh_rewardall_ppo)/len(eh_rewardall_ppo),
                        sum(eh_rewardall_greedy)/len(eh_rewardall_greedy),
                        sum(eh_rewardall_random)/len(eh_rewardall_random)]

print("=================================================================================")
print(f"Average Harvested Energy (DDPG): {avg_harvested_energy[0]}")
print(f"Average Harvested Energy (PER-DDPG): {avg_harvested_energy[1]}")
print(f"Average Harvested Energy (CER-DDPG): {avg_harvested_energy[2]}")
print(f"Average Harvested Energy (TD3): {avg_harvested_energy[3]}")
print(f"Average Harvested Energy (PPO): {avg_harvested_energy[4]}")
print(f"Average Harvested Energy (Greedy): {avg_harvested_energy[5]}")
print(f"Average Harvested Energy (Random): {avg_harvested_energy[6]}")
print("==================================================================================")
print("=================================================================================")
print(f"Average Episodic Rewards (DDPG): {avg_rewards[0]}")
print(f"Average Episodic Rewards (PER-DDPG): {avg_rewards[1]}")
print(f"Average Episodic Rewards (CER-DDPG): {avg_rewards[2]}")
print(f"Average Episodic Rewards (TD3): {avg_rewards[3]}")
print(f"Average Episodic Rewards (PPO): {avg_rewards[4]}")
print(f"Average Episodic Rewards (Greedy): {avg_rewards[5]}")
print(f"Average Episodic Rewards (Random): {avg_rewards[6]}")
print("==================================================================================")


######### WRITING DATA TO FILES ##########
'''
rewards_save=[ep_rewardall_ddpg, ep_rewardall_per, ep_rewardall_cer, ep_rewardall_td3, ep_rewardall_ppo, ep_rewardall_greedy, ep_rewardall_random]
file = open('reward_arrays_r1.txt','a')
for item in rewards_save:
      file.write(str(item))
      file.write("\n")
file.close()

harvested_energy_save=[eh_rewardall_ddpg, eh_rewardall_per, eh_rewardall_cer, eh_rewardall_td3, eh_rewardall_ppo, eh_rewardall_greedy, eh_rewardall_random]
file=open('EH_arrays_r2.txt','a')
for item in harvested_energy_save:
      file.write(str(item))
      file.write("\n")
file.close()

file=open('avg_EH_r3_v2.txt','a')
for item in avg_harvested_energy:
      file.write(str(item))
      file.write("\n")
file.close()

file=open('avg_rewards_r3_v2.txt','a')
for item in avg_rewards:
      file.write(str(item))
      file.write("\n")
file.close()
'''
file=open('avg_alphan_arrays.txt','a')
for item in avg_alphan:
      file.write(str(item))
      file.write("\n")
file.close()

'''
########### PLOTTING FIGURES ###############
fig, ax = plt.subplots()
ax.plot(ep_rewardall_ddpg, "^-", label='DDPG: rewards')
ax.plot(ep_rewardall_per, "*-", label='PER: rewards')
ax.plot(ep_rewardall_cer, "x-", label='CER: rewards')
ax.plot(ep_rewardall_td3, "s-", label='TD3: rewards')
ax.plot(ep_rewardall_ppo, "d-", label='PPO: rewards')
ax.plot(ep_rewardall_greedy, "+:", label='Greedy: rewards')
ax.plot(ep_rewardall_random, "o--", label='Random: rewards')
ax.set_xlabel("Episodes")
ax.set_ylabel(" Epsiodic Rewards")
ax.legend()
ax.margins(x=0)
ax.set_xlim(1, MAX_EPISODES-1)
ax.grid(which = "both", axis='y', linestyle=':', color='gray', linewidth=0.5)
ax.minorticks_on()
ax.tick_params(which = "minor", bottom = False, left = False)

# Define a custom tick formatter function
from matplotlib.ticker import FuncFormatter
fig2, ax = plt.subplots()
ax.plot(eh_rewardall_ddpg, "^-", label='DDPG: EH')
ax.plot(eh_rewardall_per, "*-", label='PER: EH')
ax.plot(eh_rewardall_cer, "x-", label='CER: EH')
ax.plot(eh_rewardall_td3, "s-", label='TD3: EH')
ax.plot(eh_rewardall_ppo, "d-", label='PPO: EH')
ax.plot(eh_rewardall_greedy, "+:", label='Greedy: EH')
ax.plot(eh_rewardall_random, "o--", label='Random: EH')
ax.set_xlabel("Episodes")
ax.set_ylabel(" Energy Harvested (J)")
ax.legend()
ax.margins(x=0)
ax.set_xlim(1, MAX_EPISODES-1)
ax.set_yscale('log')
ax.grid(which = "both", axis='y', linestyle=':', color='gray', linewidth=0.5)
ax.minorticks_on()
ax.tick_params(which = "minor", bottom = False, left = False)

plt.show()

'''
