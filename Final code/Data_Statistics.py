#Final version Dec 13 2019
import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

#This script will take any two excel files containing features and classes
#and read you the important information in human-compatable form.

#Dictionary of classes and manufacturers to read from
VEHICLECLASS_DICT = {"EV":0, "Hybrid":1, "Sedan":2, "SUV":3, "Pickup":4, "Bus":5, "Commercial":6}
VEHICLEMANUFACTURER_DICT = {"Toyota":0, "Volkswagen":1, "Ford":2, "Honda":3, "Nissan":4, "Hyundai":5, "Chevy":6, "Kia":7, "Mercedes":8, "BMW":9, "Audi":10, "Jeep":11, "Mazda":12, "Mitsubishi":13, "Buick":14, "Subaru":15, "Suzuki":16, "Lexus":17, "Volvo":18, "GMC":19, "Dodge":20, "Cadillac":21, "LandRover":22, "Mini":23, "Scion":24, "Tesla":25, "Chrysler":26, "Marguerite":27, "Acura":28, "Infiniti":29, "Jaguar":30, "Other":31}    

#Function that reads from the dictionaries
def get_key(val, my_dict): 
    for key, value in my_dict.items(): 
         if val == value: 
             return key 
    return "key doesn't exist"

#Import classes and features
Y_train_path = './car_data_Y_audio50.csv'
Y_data = np.asarray(pd.read_csv(Y_train_path, header= None)) 
X_train_path = './car_data_X_audio50.csv'
X_data = np.asarray(pd.read_csv(X_train_path, header= None))

#Separate class indicators into classes (i.e. bus) and manufacturer(i.e. BMW)
Classes = (Counter(Y_data[:,0]))
Manufacturers = (Counter(Y_data[:,1]))

#Tally and print distribution of classes
print('\n Displaying class distribution!')
class_index = 0
while class_index < len(VEHICLECLASS_DICT):
    if Classes[class_index] > 1:
        print(' There are', Classes[class_index], get_key(class_index, VEHICLECLASS_DICT))
    class_index += 1
    
#Tally and print distribution of manufacturers
print('\n Displaying manufacturer distribution!')
manu_index = 0
while manu_index < len(VEHICLEMANUFACTURER_DICT):
    if Manufacturers[manu_index] > 1:
        print(' There are', Manufacturers[manu_index], get_key(manu_index, VEHICLEMANUFACTURER_DICT))
    manu_index += 1

#For plotting all the data
colors = ['lime','dodgerblue','darkorange','blue','fuchsia', 'black', 'red']
markers = ["$To$" , "$Vo$" , "$F$" , "$H$" , "$N$" , "$Hy$", "$C$", "$K$", "$M$", "$B$", "$A$", "$J$", "$M$", "$Mi$", "$B$", "$S$", "$Su$", "$L$", "$V$", "$G$", "$D$", "$C$", "$LR$", "$M$", "$S$", "$T$", "$Ch$", "$Ma$", "$Ac$", "$I$", "$J$", "*"]

f = plt.figure(figsize=(4, 4), dpi=180)
for j in range(Y_data.shape[0]):
        mi = markers[int(Y_data[j,1])]
        ci = colors[int(Y_data[j,0])]
        #Can change the below if statement to only plot certain classes
        if Y_data[j,0] == 0 or Y_data[j,0] == 1 or Y_data[j,0] == 2 or Y_data[j,0] == 3 or Y_data[j,0] == 4 or Y_data[j,0] == 5 or Y_data[j,0] == 6:
            plt.scatter(X_data[j,0],X_data[j,300], marker = mi, color = ci)
    
legend_elements = [
                    Line2D([0], [0], marker='o', color='w', label='EV', markerfacecolor='lime', markersize=8),
                    Line2D([0], [0], marker='o', color='w', label='Hybrid', markerfacecolor='dodgerblue', markersize=8),
                    Line2D([0], [0], marker='o', color='w', label='Sedan', markerfacecolor='darkorange', markersize=8),
                    Line2D([0], [0], marker='o', color='w', label='SUV', markerfacecolor='blue', markersize=8),
                    Line2D([0], [0], marker='o', color='w', label='Pickup', markerfacecolor='fuchsia', markersize=8),
                    Line2D([0], [0], marker='o', color='w', label='Bus', markerfacecolor='black', markersize=8),
                    Line2D([0], [0], marker='o', color='w', label='Commercial', markerfacecolor='red', markersize=8)
                    ]
    
plt.xlabel('Vehicle Dynamic Range')
plt.ylabel('$FR_{300}$')
ax = plt.gca()
ax.set_yscale('log')
ax.legend(handles=legend_elements, prop={'size': 6}, frameon = False)
ax.set_yticklabels([])
ax.set_xticklabels([])
ax.tick_params(axis=u'both', which=u'both',length=0)
