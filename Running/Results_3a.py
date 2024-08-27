import matplotlib.pyplot as plt

avg_EH_file = open('C:/Users/PMLS/Documents/PythonVENV/avg_EH_r3_v2.txt')
avg_EH_data = avg_EH_file.readlines()
avg_EH_data = [float(i.strip('[]\n')) for i in avg_EH_data]
#avg_EH_data = [float(i) for i in avg_EH_data]
#avg_EH_data = [round(EH_data, 2) for EH_data in avg_EH_data]

EH_EGC_array_ddpg = [avg_EH_data[0],avg_EH_data[21],avg_EH_data[42],avg_EH_data[63],avg_EH_data[84]]
EH_EGC_array_per = [avg_EH_data[1],avg_EH_data[22],avg_EH_data[43],avg_EH_data[64],avg_EH_data[85]]
EH_EGC_array_cer = [avg_EH_data[2],avg_EH_data[23],avg_EH_data[44],avg_EH_data[65],avg_EH_data[86]]
EH_EGC_array_td3 = [avg_EH_data[3],avg_EH_data[24],avg_EH_data[45],avg_EH_data[66],avg_EH_data[87]]
EH_EGC_array_ppo = [avg_EH_data[4],avg_EH_data[25],avg_EH_data[46],avg_EH_data[67],avg_EH_data[88]]
EH_EGC_array_greedy = [avg_EH_data[5],avg_EH_data[26],avg_EH_data[47],avg_EH_data[68],avg_EH_data[89]]
EH_EGC_array_random = [avg_EH_data[6],avg_EH_data[27],avg_EH_data[48],avg_EH_data[69],avg_EH_data[90]]

EH_MRC_array_ddpg = [avg_EH_data[7],avg_EH_data[28],avg_EH_data[49],avg_EH_data[70],avg_EH_data[91]]
EH_MRC_array_per = [avg_EH_data[8],avg_EH_data[29],avg_EH_data[50],avg_EH_data[71],avg_EH_data[92]]
EH_MRC_array_cer = [avg_EH_data[9],avg_EH_data[30],avg_EH_data[51],avg_EH_data[72],avg_EH_data[93]]
EH_MRC_array_td3 = [avg_EH_data[10],avg_EH_data[31],avg_EH_data[52],avg_EH_data[73],avg_EH_data[94]]
EH_MRC_array_ppo = [avg_EH_data[11],avg_EH_data[32],avg_EH_data[53],avg_EH_data[74],avg_EH_data[95]]
EH_MRC_array_greedy = [avg_EH_data[12],avg_EH_data[33],avg_EH_data[54],avg_EH_data[75],avg_EH_data[96]]
EH_MRC_array_random = [avg_EH_data[13],avg_EH_data[34],avg_EH_data[55],avg_EH_data[76],avg_EH_data[97]]

EH_SC_array_ddpg = [avg_EH_data[14],avg_EH_data[35],avg_EH_data[56],avg_EH_data[77],avg_EH_data[98]]
EH_SC_array_per = [avg_EH_data[15],avg_EH_data[36],avg_EH_data[57],avg_EH_data[78],avg_EH_data[99]]
EH_SC_array_cer = [avg_EH_data[16],avg_EH_data[37],avg_EH_data[58],avg_EH_data[79],avg_EH_data[100]]
EH_SC_array_td3 = [avg_EH_data[17],avg_EH_data[38],avg_EH_data[59],avg_EH_data[80],avg_EH_data[101]]
EH_SC_array_ppo = [avg_EH_data[18],avg_EH_data[39],avg_EH_data[60],avg_EH_data[81],avg_EH_data[102]]
EH_SC_array_greedy = [avg_EH_data[19],avg_EH_data[40],avg_EH_data[61],avg_EH_data[82],avg_EH_data[103]]
EH_SC_array_random = [avg_EH_data[20],avg_EH_data[41],avg_EH_data[62],avg_EH_data[83],avg_EH_data[104]]

print(EH_EGC_array_ddpg)

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
ax.set_ylabel("Energy Harvested (J)")
ax.set_title("EGC")
ax.legend(['DDPG', 'PER', 'CER', 'TD3', 'PPO', 'Greedy', 'Random'])
ax.margins(x=0)
ax.set_xlim(2, 6, 1)
ax.set_yscale('log')
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
ax.set_ylabel("Energy Harvested (J)")
ax.set_title("MRC")
ax.legend(['DDPG', 'PER', 'CER', 'TD3', 'PPO', 'Greedy', 'Random'])
ax.margins(x=0)
ax.set_xlim(2, 6, 1)
ax.set_yscale('log')
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
ax.set_ylabel("Energy Harvested (J)")
ax.set_title("SC")
ax.legend(['DDPG', 'PER', 'CER', 'TD3', 'PPO', 'Greedy', 'Random'])
ax.margins(x=0)
ax.set_xlim(2, 6, 1)
ax.set_yscale('log')
ax.grid(which = "both", axis='y', linestyle=':', color='gray', linewidth=0.5)
ax.minorticks_on()
ax.tick_params(which = "minor", bottom = False, left = False)
 
plt.show()