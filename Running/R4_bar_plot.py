import numpy as np 
import matplotlib.pyplot as plt 
 
# set width of bar 
barWidth = 0.25
fig = plt.subplots(figsize =(12, 8)) 

alphan_file = open('C:/Users/PMLS/Documents/PythonVENV/R4_avg_alphan_arrays_td3.txt')
avg_alphan_data = alphan_file.readlines()
avg_alphan_data = [round(float(s.strip('[]\n')), 3) for s in avg_alphan_data]
print(avg_alphan_data)
#avg_alphan_data = [float(i.strip('[]\n')) for i in avg_alphan_data]
 
# set height of bar 
alphan = [avg_alphan_data[0], avg_alphan_data[1], avg_alphan_data[2], avg_alphan_data[3]]
one_minus_alphan = [1-avg_alphan_data[0], 1-avg_alphan_data[1], 1-avg_alphan_data[2], 1-avg_alphan_data[3]]
 
# Set position of bar on X axis 
br1 = np.arange(len(alphan)) 
br2 = [x + barWidth for x in br1] 

# Make the plot
plt.bar(br1, alphan, color ='green', width = barWidth, 
        edgecolor ='g', label ='SN Transmission Time Fraction') 
plt.bar(br2, one_minus_alphan, color ='red', width = barWidth, 
        edgecolor ='r', label ='SN Harvesting Time Fraction') 

 
# Adding Xticks 
plt.xlabel('Diversity Technique', fontweight ='bold', fontsize = 15) 
plt.ylabel('Fraction of Time Slot', fontweight ='bold', fontsize = 15) 
plt.xticks([r + barWidth for r in range(len(alphan))], 
        ['No Diversity', 'EGC', 'MRC', 'SC'])
 
plt.legend()
plt.show() 