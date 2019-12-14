# December 6 2019

from scipy.io.wavfile import read
from scipy.fftpack import fft
import matplotlib.pyplot as plt
import os
import numpy as np
import re
import numpy.fft as fft
import pandas as pd
from matplotlib.lines import Line2D
import librosa
import wave
import seaborn


def Class_Manu_Audio(vehicle_filename, VEHICLETYPE_DICT, VEHICLEMANUFACTURER_DICT):
    framerate, vehicle_audio = read(vehicle_filename);
    vehicle_audio = vehicle_audio[:, 0]
    for key, value in VEHICLEMANUFACTURER_DICT.items():
        if key in vehicle_filename:
            vehicle_manu = key
    for key, value in VEHICLETYPE_DICT.items():
        if key in vehicle_filename:
            vehicle_class = key
    print(vehicle_class, vehicle_manu)
    return vehicle_audio, vehicle_class, vehicle_manu, VEHICLECLASS_DICT[vehicle_class], VEHICLEMANUFACTURER_DICT[
        vehicle_manu], framerate


def DR_MaxInt_Var(vehicle_audio):
    vehicle_audio = np.abs(vehicle_audio)
    vehicle_audio = sorted(vehicle_audio);
    background = np.mean(vehicle_audio[1:1000]);
    max_intensity = np.mean(vehicle_audio[-20000:-1])
    dynamic_range = max_intensity - background
    variance = np.var(vehicle_audio)
    return dynamic_range, max_intensity * 10, variance * 100


def FFT_Matrix(freq, spectrum):
    spectrum = np.abs(spectrum)
    chunk_spectrum = np.zeros(25)
    chunk_matrix = np.zeros([25, 25])
    i = 0
    while i < len(chunk_spectrum):
        begin_value = i / 50;
        end_value = (i + 1) / 50
        begin_index = min(range(len(freq)), key=lambda i: abs(freq[i] - begin_value))
        end_index = min(range(len(freq)), key=lambda i: abs(freq[i] - end_value))
        sorted_chunk = sorted(spectrum[begin_index:end_index])
        sorted_chunk = np.asarray(sorted_chunk)
        chunk_value = np.mean(sorted_chunk[-200:-1])
        chunk_spectrum[i] = chunk_value
        i += 1
    for i in range(len(chunk_spectrum)):
        for j in range(len(chunk_spectrum)):
            chunk_matrix[i, j] = chunk_spectrum[i] / chunk_spectrum[j]
    chunk_matrix = chunk_matrix.reshape(25 * 25)
    return chunk_matrix


def make_audio_same_length(vehicle_audio, max_length):
    new_clip = np.zeros(max_length)
    vehicle_length = vehicle_audio.shape[0]
    length_front = int((max_length - vehicle_length) / 2)
    background = np.mean(vehicle_audio[1:1000])
    new_clip[0:length_front] = np.ones(length_front) * background
    new_clip[length_front: (length_front + vehicle_length)] = vehicle_audio
    new_clip[(length_front + vehicle_length):] = np.ones(max_length - length_front - vehicle_length) * background
    return new_clip


def addNoise(vehicle_audio, noise_factor):
    noise = np.random.randn(len(vehicle_audio))
    augmented_data = vehicle_audio + noise_factor * noise
    # Cast back to same data type
    augmented_data = augmented_data.astype(type(vehicle_audio[0]))
    return augmented_data


def changePitch(vehicle_audio, sampling_rate, pitch_factor):
    return librosa.effects.pitch_shift(vehicle_audio, sampling_rate, pitch_factor)


VEHICLECLASS_DICT = {"EV": 0, "Hybrid": 1, "Sedan": 2, "SUV": 3, "Pickup": 4, "Bus": 5, "Commercial": 6}
VEHICLEMANUFACTURER_DICT = {"Toyota": 0, "Volkswagen": 1, "Ford": 2, "Honda": 3, "Nissan": 4, "Hyundai": 5, "Chevy": 6,
                            "Kia": 7, "Mercedes": 8, "BMW": 9, "Audi": 10, "Jeep": 11, "Mazda": 12, "Mitsubishi": 13,
                            "Buick": 14, "Subaru": 15, "Suzuki": 16, "Lexus": 17, "Volvo": 18, "GMC": 19, "Dodge": 20,
                            "Cadillac": 21, "LandRover": 22, "Mini": 23, "Scion": 24, "Tesla": 25, "Chrysler": 26,
                            "Marguerite": 27, "Acura": 28, "Infiniti": 29, "Jaguar": 30, "Other": 31}

directory = os.fsencode(r"C:\Users\ge820\Desktop\229\project\data_allwav")  # For importing data

data_vector = []  # initialize vector to store all data (num_cars, 3). Records max intensity, average intensity, and class index from dict. Can add other stuff later
total_length_collection = []
vehicle_audio_collection = []
n_car_processed = 0

car_num_counter = 0
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".wav"):
        vehicle_audio, vehicle_class_name, vehicle_manu, vehicle_class_index, vehicle_manu_index, framerate = Class_Manu_Audio(
            os.path.join(filename), VEHICLECLASS_DICT, VEHICLEMANUFACTURER_DICT)

        audio_list_original_plus_augmented = []
        # audio_list_original_plus_augmented.append(vehicle_audio)

        '''
        Noise introduction
        For each car that is non-Sedan and non-SUV, create 3 more datapoints
        '''
        noise_factor_list = [0.03, 0.02, 0.025]
        if vehicle_class_index != 2 and vehicle_class_index != 3:
            if vehicle_class_index != 1:
                for noise_factor in noise_factor_list:
                    vehicle_audio_noised = addNoise(vehicle_audio, noise_factor)
                    audio_list_original_plus_augmented.append(vehicle_audio_noised)
                    # plt.plot(vehicle_audio_noised, 'lightsalmon')
                    # plt.plot(vehicle_audio, 'lightskyblue')
                    fig, axs = plt.subplots(2, figsize=(3.5, 2.5), dpi=180)
                    axs[0].plot(vehicle_audio_noised, 'lightsalmon')
                    axs[1].plot(vehicle_audio, 'lightskyblue')
                    plt.xlabel("Time")
                    plt.ylabel("Amplitude")
                    # plt.xlim([0, 0.5]);
                    # plt.ylim([0,50])
                    for ax in axs:
                       ax.set_xticks([])
                       ax.set_yticks([])
                    plt.show()

            elif vehicle_class_index == 1:
                noise_factor = noise_factor_list[0]
                vehicle_audio_noised = addNoise(vehicle_audio, noise_factor)
                audio_list_original_plus_augmented.append(vehicle_audio_noised)


        '''
        Pitch varied
        For each car that is non-Sedan and non-SUV, create 2 more datapoints 
        '''
        if vehicle_class_index != 2 and vehicle_class_index != 3:
            if vehicle_class_index != 1:
                vehicle_audio_pitched = changePitch(np.asfortranarray(vehicle_audio), framerate, 1)
                vehicle_audio_pitched_2 = changePitch(np.asfortranarray(vehicle_audio), framerate, 4)
                audio_list_original_plus_augmented.append(vehicle_audio_pitched)
                audio_list_original_plus_augmented.append(vehicle_audio_pitched_2)
                # plt.plot(vehicle_audio)
                # plt.plot(vehicle_audio_pitched)
                # plt.plot(vehicle_audio_pitched_2)
                # plt.show()
            elif vehicle_class_index == 1:
                vehicle_audio_pitched = changePitch(np.asfortranarray(vehicle_audio), framerate, 1)
                audio_list_original_plus_augmented.append(vehicle_audio_pitched)


        print("audio_list_len = ", len(audio_list_original_plus_augmented))

        for audio in audio_list_original_plus_augmented:
            # Emmett's original code for the three params and FFT
            vehicle_dynamic_range, max_intensity, vehicle_variance = DR_MaxInt_Var(audio)
            spectrum = fft.fft(audio);
            freq = np.abs(fft.fftfreq(len(spectrum)));
            vehicle_maxminfft = FFT_Matrix(freq, spectrum)
            vehicle_maxminfft = list(vehicle_maxminfft)
            vehicle_data = (
            [vehicle_class_index, vehicle_manu_index, vehicle_dynamic_range, max_intensity, vehicle_variance])
            vehicle_data.extend(vehicle_maxminfft)
            data_vector.append(vehicle_data)
            # my code for generating audio50
            total_length_collection.append(audio.shape[0])
            vehicle_audio_collection.append(audio)
            n_car_processed += 1
            print("number of cars FFTed = ", n_car_processed)

        car_num_counter += 1
        print(car_num_counter)
        continue

print("for loop finished!!!!!")
print("vehicle_audio_collection length = ", len(vehicle_audio_collection))
print("data_vector length", len(data_vector))

'''
divide the raw audio data into 50 groups
'''
max_length = np.max(total_length_collection)  # this num = 396568
time_resolution = 50
timeunit_per_res = int(max_length / time_resolution)

# new_clip_collection = np.zeros((len(vehicle_audio_collection), max_length))
new_clip_collection = np.zeros((len(vehicle_audio_collection), time_resolution))

for idx, clip in enumerate(vehicle_audio_collection):
    # plt.plot(clip, 'b')
    new_clip = make_audio_same_length(clip, max_length)
    # plt.plot(new_clip, 'r')
    # plt.show()
    # new_just_for_plot = []
    new_clip_fit_resolution = []
    for i in range(time_resolution):
        max_in_timeunit = np.max(new_clip[i * timeunit_per_res: (i + 1) * timeunit_per_res])
        new_clip_fit_resolution.append(max_in_timeunit)
        # new_just_for_plot += timeunit_per_res * [max_in_timeunit]
    new_clip_collection[idx] = np.asarray(new_clip_fit_resolution)
    data_vector[idx].extend(new_clip_fit_resolution)
    # plt.plot(new_clip, 'b')
    # plt.plot(new_just_for_plot, 'r')
    # plt.show()

data_vector = np.asarray(data_vector)
print("final shape = ", data_vector.shape)

# colors = ['r','b','lime','gold','fuchsia', 'black', 'dodgerblue']
# markers = ["$To$" , "$Vo$" , "$F$" , "$H$" , "$N$" , "$Hy$", "$C$", "$K$", "$M$", "$B$", "$A$", "$J$", "$M$", "$Mi$", "$B$", "$S$", "$Su$", "$L$", "$V$", "$G$", "$D$", "$C$", "$LR$", "$M$", "$S$", "$T$", "$Ch$", "$Ma$", "$Ac$", "$I$", "$J$", "*"]
#
# f = plt.figure(figsize=(3.5, 2.5), dpi=180)
# for j in range(car_num_counter):
#     mi = markers[int(data_vector[j,1])]
#     ci = colors[int(data_vector[j,0])]
#     plt.scatter(data_vector[j,0*25+3+5],data_vector[j,20*25+15+5], marker = mi, color = ci)
#
# legend_elements = [
#                    Line2D([0], [0], marker='o', color='w', label='EV', markerfacecolor='g', markersize=8),
#                    Line2D([0], [0], marker='o', color='w', label='Hybrid', markerfacecolor='b', markersize=8),
#                    Line2D([0], [0], marker='o', color='w', label='Sedan', markerfacecolor='lime', markersize=8),
#                    Line2D([0], [0], marker='o', color='w', label='SUV', markerfacecolor='gold', markersize=8),
#                    Line2D([0], [0], marker='o', color='w', label='Pickup', markerfacecolor='fuchsia', markersize=8),
#                    Line2D([0], [0], marker='o', color='w', label='Bus', markerfacecolor='black', markersize=8),
#                    Line2D([0], [0], marker='o', color='w', label='Commercial', markerfacecolor='dodgerblue', markersize=8)
#                   ]
#
# plt.xlabel('Vehicle Dynamic Range')
# plt.ylabel('Fourier Ratio')
# ax = plt.gca()
# ax.set_yscale('log')
# ax.legend(handles=legend_elements, prop={'size': 6}, frameon = False)
#
#
df_X = pd.DataFrame(data_vector[:, [x for x in range(2, 680)]])  # range(2,25*25+5)
df_X.to_csv("car_data_X_augmented.csv", index=False)

df_Y = pd.DataFrame(data_vector[:, range(0, 2)])  # columns=["vehicle_class_index"]
df_Y.to_csv("car_data_Y_augmented.csv", index=False)

