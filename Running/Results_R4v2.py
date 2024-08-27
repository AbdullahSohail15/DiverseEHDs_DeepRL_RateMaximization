import numpy as np
import matplotlib.pyplot as plt

# Data
avg_alphan_file = open('C:/Users/PMLS/Documents/PythonVENV/avg_alphan_arrays.txt')
avg_alphan_data = avg_alphan_file.readlines()
avg_alphan_data = [round(float(i), 5) for i in avg_alphan_data]
print(type(avg_alphan_data[0]))

categories = ['DDPG', 'PER-DDPG', 'CER_DDPG','TD3', 'PPO', 'Random', 'Greedy']
labels = ['No Diversity', 'EGC', 'MRC', 'SC']
values1 = np.array([[avg_alphan_data[0], avg_alphan_data[7], avg_alphan_data[14], avg_alphan_data[21]], 
                    [avg_alphan_data[1], avg_alphan_data[8], avg_alphan_data[15], avg_alphan_data[22]], 
                    [avg_alphan_data[2], avg_alphan_data[9], avg_alphan_data[16], avg_alphan_data[23]],
                    [avg_alphan_data[3], avg_alphan_data[10], avg_alphan_data[17], avg_alphan_data[24]], 
                    [avg_alphan_data[4], avg_alphan_data[11], avg_alphan_data[18], avg_alphan_data[25]], 
                    [avg_alphan_data[5], avg_alphan_data[12], avg_alphan_data[19], avg_alphan_data[26]], 
                    [avg_alphan_data[6], avg_alphan_data[13], avg_alphan_data[20], avg_alphan_data[27]]])  # Group 1 data

values2 = np.array([[1-avg_alphan_data[0], 1-avg_alphan_data[7], 1-avg_alphan_data[14], 1-avg_alphan_data[21]], 
                    [1-avg_alphan_data[1], 1-avg_alphan_data[8], 1-avg_alphan_data[15], 1-avg_alphan_data[22]], 
                    [1-avg_alphan_data[2], 1-avg_alphan_data[9], 1-avg_alphan_data[16], 1-avg_alphan_data[23]],
                    [1-avg_alphan_data[3], 1-avg_alphan_data[10], 1-avg_alphan_data[17], 1-avg_alphan_data[24]], 
                    [1-avg_alphan_data[4], 1-avg_alphan_data[11], 1-avg_alphan_data[18], 1-avg_alphan_data[25]], 
                    [1-avg_alphan_data[5], 1-avg_alphan_data[12], 1-avg_alphan_data[19], 1-avg_alphan_data[26]], 
                    [1-avg_alphan_data[6], 1-avg_alphan_data[13], 1-avg_alphan_data[20], 1-avg_alphan_data[27]]])  # Group 1 data

# Plot
fig, ax = plt.subplots()

width = 0.08  # Width of bars
ind = np.arange(len(labels))  # x locations for groups

#FD1F4A
#FBBD0D

# Plot bars for each group
p1 = ax.bar(ind, values1[0], width, color = '#FF0000', label='SN Transmission Time Fraction',edgecolor='black', linewidth = 0.5)
p2 = ax.bar(ind, values2[0], width, color = 'orange', bottom=values1[0], label='SN Harvesting Time Fraction', edgecolor='black', linewidth = 0.5)

# Add more bars for the rest of the groups
for i in range(1, len(categories)):
    ax.bar(ind + ((width+0.04) * i), values1[i], width, color = '#FF0000', edgecolor='black', linewidth = 0.5)
    ax.bar(ind + ((width+0.04) * i), values2[i], width, color = 'orange', bottom=values1[i], edgecolor='black', linewidth = 0.5)

# Add labels, title, and legend
ax.set_xticks(ind + width * (len(categories) - 1) / 2)
ax.set_xticklabels(labels)
ax.set_ylabel('Fraction of Time Slot')
ax.legend(loc = 'upper right')
#ax.get_tightbbox()
#plt.tight_layout()
plt.savefig('C:/Users/PMLS/Documents/PythonVENV/R4_FIG.eps', format='eps')
plt.show()
