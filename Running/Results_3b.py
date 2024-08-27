import matplotlib.pyplot as plt

avg_rewards_file = open('C:/Users/PMLS/Documents/PythonVENV/avg_rewards_r3_v2.txt')
avg_reward_data = avg_rewards_file.readlines()
avg_reward_data = [float(i.strip('[]\n')) for i in avg_reward_data]

EH_EGC_array_ddpg = [avg_reward_data[0],avg_reward_data[21],avg_reward_data[42],avg_reward_data[63],avg_reward_data[84]]
EH_EGC_array_per = [avg_reward_data[1],avg_reward_data[22],avg_reward_data[43],avg_reward_data[64],avg_reward_data[85]]
EH_EGC_array_cer = [avg_reward_data[2],avg_reward_data[23],avg_reward_data[44],avg_reward_data[65],avg_reward_data[86]]
EH_EGC_array_td3 = [avg_reward_data[3],avg_reward_data[24],avg_reward_data[45],avg_reward_data[66],avg_reward_data[87]]
EH_EGC_array_ppo = [avg_reward_data[4],avg_reward_data[25],avg_reward_data[46],avg_reward_data[67],avg_reward_data[88]]
EH_EGC_array_greedy = [avg_reward_data[5],avg_reward_data[26],avg_reward_data[47],avg_reward_data[68],avg_reward_data[89]]
EH_EGC_array_random = [avg_reward_data[6],avg_reward_data[27],avg_reward_data[48],avg_reward_data[69],avg_reward_data[90]]

EH_MRC_array_ddpg = [avg_reward_data[7],avg_reward_data[28],avg_reward_data[49],avg_reward_data[70],avg_reward_data[91]]
EH_MRC_array_per = [avg_reward_data[8],avg_reward_data[29],avg_reward_data[50],avg_reward_data[71],avg_reward_data[92]]
EH_MRC_array_cer = [avg_reward_data[9],avg_reward_data[30],avg_reward_data[51],avg_reward_data[72],avg_reward_data[93]]
EH_MRC_array_td3 = [avg_reward_data[10],avg_reward_data[31],avg_reward_data[52],avg_reward_data[73],avg_reward_data[94]]
EH_MRC_array_ppo = [avg_reward_data[11],avg_reward_data[32],avg_reward_data[53],avg_reward_data[74],avg_reward_data[95]]
EH_MRC_array_greedy = [avg_reward_data[12],avg_reward_data[33],avg_reward_data[54],avg_reward_data[75],avg_reward_data[96]]
EH_MRC_array_random = [avg_reward_data[13],avg_reward_data[34],avg_reward_data[55],avg_reward_data[76],avg_reward_data[97]]

EH_SC_array_ddpg = [avg_reward_data[14],avg_reward_data[35],avg_reward_data[56],avg_reward_data[77],avg_reward_data[98]]
EH_SC_array_per = [avg_reward_data[15],avg_reward_data[36],avg_reward_data[57],avg_reward_data[78],avg_reward_data[99]]
EH_SC_array_cer = [avg_reward_data[16],avg_reward_data[37],avg_reward_data[58],avg_reward_data[79],avg_reward_data[100]]
EH_SC_array_td3 = [avg_reward_data[17],avg_reward_data[38],avg_reward_data[59],avg_reward_data[80],avg_reward_data[101]]
EH_SC_array_ppo = [avg_reward_data[18],avg_reward_data[39],avg_reward_data[60],avg_reward_data[81],avg_reward_data[102]]
EH_SC_array_greedy = [avg_reward_data[19],avg_reward_data[40],avg_reward_data[61],avg_reward_data[82],avg_reward_data[103]]
EH_SC_array_random = [avg_reward_data[20],avg_reward_data[41],avg_reward_data[62],avg_reward_data[83],avg_reward_data[104]]


Num_Antennas = [2, 3, 4, 5, 6]

fig, ax = plt.subplots()

ax.plot(Num_Antennas, EH_EGC_array_ddpg, "^-", label='DDPG: EH')
ax.plot(Num_Antennas, EH_EGC_array_per, "*-", label='PER: EH')
ax.plot(Num_Antennas, EH_EGC_array_cer, "x-", label='PER: EH')
ax.plot(Num_Antennas, EH_EGC_array_td3, "s-", label='TD3: EH')
ax.plot(Num_Antennas, EH_EGC_array_ppo, "d-", label='PPO: EH')
ax.plot(Num_Antennas, EH_EGC_array_greedy, "+:", label='Greedy: EH')
ax.plot(Num_Antennas, EH_EGC_array_random, "o--", label='Random: EH')

ax.set_xlabel("Num. of SN Antennas")
ax.set_ylabel("Episodic Rewards")
ax.set_title("EGC")
ax.legend(['DDPG', 'PER', 'CER', 'TD3', 'PPO', 'Greedy', 'Random'])
ax.margins(x=0)
ax.set_xlim(2, 6, 1)
ax.grid(which = "both", axis='y', linestyle=':', color='gray', linewidth=0.5)
ax.minorticks_on()
ax.tick_params(which = "minor", bottom = False, left = False)


fig2, ax = plt.subplots()
ax.plot(Num_Antennas, EH_MRC_array_ddpg, "^-", label='DDPG: EH')
ax.plot(Num_Antennas, EH_MRC_array_per, "*-", label='PER: EH')
ax.plot(Num_Antennas, EH_MRC_array_cer, "x-", label='PER: EH')
ax.plot(Num_Antennas, EH_MRC_array_td3, "s-", label='TD3: EH')
ax.plot(Num_Antennas, EH_MRC_array_ppo, "d-", label='PPO: EH')
ax.plot(Num_Antennas, EH_MRC_array_greedy, "+:", label='Greedy: EH')
ax.plot(Num_Antennas, EH_MRC_array_random, "o--", label='Random: EH')
ax.set_xlabel("Num. of SN Antennas")
ax.set_ylabel("Episodic Rewards")
ax.set_title("MRC")
ax.legend(['DDPG', 'PER', 'CER', 'TD3', 'PPO', 'Greedy', 'Random'])
ax.margins(x=0)
ax.set_xlim(2, 6, 1)
ax.grid(which = "both", axis='y', linestyle=':', color='gray', linewidth=0.5)
ax.minorticks_on()
ax.tick_params(which = "minor", bottom = False, left = False)


fig3, ax = plt.subplots()
ax.plot(Num_Antennas, EH_SC_array_ddpg, "^-", label='DDPG: EH')
ax.plot(Num_Antennas, EH_SC_array_per, "*-", label='PER: EH')
ax.plot(Num_Antennas, EH_SC_array_cer, "x-", label='PER: EH')
ax.plot(Num_Antennas, EH_SC_array_td3, "s-", label='TD3: EH')
ax.plot(Num_Antennas, EH_SC_array_ppo, "d-", label='PPO: EH')
ax.plot(Num_Antennas, EH_SC_array_greedy, "+:", label='Greedy: EH')
ax.plot(Num_Antennas, EH_SC_array_random, "o--", label='Random: EH')
ax.set_xlabel("Num. of SN Antennas")
ax.set_ylabel("Episodic Rewards")
ax.set_title("SC")
ax.legend(['DDPG', 'PER', 'CER', 'TD3', 'PPO', 'Greedy', 'Random'])
ax.margins(x=0)
ax.set_xlim(2, 6, 1)
ax.grid(which = "both", axis='y', linestyle=':', color='gray', linewidth=0.5)
ax.minorticks_on()
ax.tick_params(which = "minor", bottom = False, left = False)
 
plt.show()