'''
This script contains the RNN code for the car classification for CS229 project.
'''

import torch
import pandas as pd
import torch.nn as nn
import random
import torch.optim as optim
import numpy as np
from sklearn import preprocessing


X_path = './data_allwav/car_data_X_RNN.csv'
Y_path = './data_allwav/car_data_Y_RNN.csv'

n_class = 7
n_manufacturer = 4
n_training_example = 1600

# define the demision of the parameters of RNN
n_hidden = 50
n_features = 50


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        hidden = self.tanh(hidden)
        output = self.i2o(combined)
        output = self.softmax(output)
        output = output.reshape((1, n_class))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

def loadata(X_train_path, Y_train_path):
    X_values = pd.read_csv(X_train_path, header=None)
    Y_values = pd.read_csv(Y_train_path, header=None)

    n_examples = X_values.values[1:].shape[0]
    print("n_examples  = ", n_examples)

    train_index = pd.read_csv('./dev_index_RNN.csv', header=None).values.T
    dev_index = pd.read_csv('./train_index_RNN.csv', header=None).values.T

    train_index_list = [int(i) for i in train_index[0]]
    dev_index_list = [int(i) for i in dev_index[0]]

    X_train = X_values.values[1:][train_index_list]
    Y_train = Y_values.values[1:][train_index_list]
    print(Y_train)

    Y_train_one_hot = np.zeros((len(train_index_list), n_class))
    for index, item in enumerate(Y_train):
        Y_train_one_hot[index, int(item[0])] = 1
        # if int(item[0]) == 1:
        #     Y_train_one_hot[index, 0] = 1
        # elif int(item[0]) == 2:
        #     Y_train_one_hot[index, 1] = 1
        # elif int(item[0]) == 4:
        #     Y_train_one_hot[index, 2] = 1
        # elif int(item[0]) == 5:
        #     Y_train_one_hot[index, 3] = 1
        # elif int(item[0]) == 6:
        #     Y_train_one_hot[index, 4] = 1

    X_train = torch.tensor(X_train).float()
    Y_train_class = torch.tensor(Y_train_one_hot, dtype=torch.long)
    Y_train = torch.max(Y_train_class, 1)[1]

    X_dev = X_values.values[1:][dev_index_list]
    Y_dev = Y_values.values[1:][dev_index_list]
    Y_dev_one_hot = np.zeros((len(dev_index_list), n_class))
    for index, item in enumerate(Y_dev):
        Y_dev_one_hot[index, int(item[0])] = 1
        # if int(item[0]) == 1:
        #     Y_dev_one_hot[index, 0] = 1
        # elif int(item[0]) == 2:
        #     Y_dev_one_hot[index, 1] = 1
        # elif int(item[0]) == 4:
        #     Y_dev_one_hot[index, 2] = 1
        # elif int(item[0]) == 5:
        #     Y_dev_one_hot[index, 3] = 1
        # elif int(item[0]) == 6:
        #     Y_dev_one_hot[index, 4] = 1

    X_dev = torch.tensor(X_dev).float()
    Y_dev_class = torch.tensor(Y_dev_one_hot, dtype=torch.long)
    Y_dev = torch.max(Y_dev_class, 1)[1]

    return X_train, Y_train, X_dev, Y_dev

# reshape the 2500 features into 50x50 matrix
def reshapeData(dataset):
    m, n = dataset.shape
    n_features = 50
    T_x = int(dataset.shape[1] / n_features)
    new_dateset = torch.zeros((m, n))
    for i in range(T_x):
        features_per_cyc = dataset[:, list(range(i, n, T_x))]
        new_dateset[:, i*n_features:(i+1)*n_features] = features_per_cyc
    new_dateset = torch.reshape(new_dateset, (m, T_x, 1, n_features)) # T_x = # of cycles; n_features = 50,
    return new_dateset

def randomTrainingExample(X, Y):
    m = X.shape[0]
    random_index = random.randint(0, m-1)
    selected_battery = X[random_index]
    life_selected_battery = Y[random_index]
    return selected_battery, life_selected_battery

# training process
def train(x, y):
    hidden = rnn.initHidden()
    optimizer.zero_grad()
    y = y.reshape(1)
    for i in range(x.size()[0]):
        # print("x[i] shape = ", x[i].size())
        # print("hidden shape = ", hidden.size())

        output, hidden = rnn(x[i], hidden)
    # print("output = ", output)
    # print("y = ", y)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()
    return output, loss

# output the prediction for 1 training example
def evaluate(x, y):
    hidden = rnn.initHidden()
    y = y.reshape(1)
    for i in range(x.size()[0]):
        output, hidden = rnn(x[i], hidden)
    loss = criterion(output, y)
    return output, loss

# Define evaluation matric
def calculate_accuracy(X, Y):
    m = X.shape[0]
    outputs = torch.zeros((m, n_class))
    for i in range(m):
        output, loss = evaluate(X[i], Y[i])
        outputs[i,:] = output
    Y_pred = torch.max(outputs, 1)[1]
    Y_pred = Y_pred.float()
    print("Y_pred = ", Y_pred)
    print("Y = ", Y)
    num_correct = torch.sum(Y.float() == Y_pred)
    acc = (num_correct * 100.0 / len(Y)).item()
    return acc

X_train, Y_train, X_dev, Y_dev = loadata(X_path, Y_path)

# normalizing the data
X_train_norm = torch.tensor(preprocessing.scale(X_train, axis=0, with_mean=True, with_std=True, copy=True)).float()
X_dev_norm = torch.tensor(preprocessing.scale(X_dev, axis=0, with_mean=True, with_std=True, copy=True)).float()

# reshape the data
X_train_norm_reshaped = reshapeData(X_train_norm)
X_dev_norm_reshaped = reshapeData(X_dev_norm)
# print("X_train_norm_reshaped size", X_train_norm_reshaped.size() )
# print("X_dev_norm_reshaped size", X_dev_norm_reshaped.size() )

n_iters = 30000
print_every = 100
plot_every = 10

# Keep track of losses for plotting
all_losses_diffLR = {}

learning_rate = [0.1, 0.03, 0.01, 0.003, 0.001]

for j in range(len(learning_rate)):
    train_current_loss = 0
    train_all_losses = []

    rnn = RNN(n_features, n_hidden, n_class)
    # Define the cost function
    criterion = nn.CrossEntropyLoss()
    # Define the optimizer
    optimizer = optim.Adam(rnn.parameters(), lr=learning_rate[j])

    for iter in range(1, n_iters + 1):
        train_selected_battery, train_battery_life = randomTrainingExample(X_train_norm_reshaped, Y_train)
        # print("train_selected_battery size", train_selected_battery.size())
        # print("train_battery_life size", train_battery_life.size())

        train_output, train_loss = train(train_selected_battery, train_battery_life)
        train_loss = train_loss.item()
        train_current_loss += train_loss  # Add current loss avg to list of losses

        if iter % plot_every == 0:
            train_all_losses.append(train_current_loss / plot_every)
            train_current_loss = 0

    all_losses_diffLR[learning_rate[j]] = train_all_losses

    training_accuracy = calculate_accuracy(X_train_norm_reshaped, Y_train)
    dev_accuracy = calculate_accuracy(X_dev_norm_reshaped, Y_dev)
    print('training accuracy with lr =  ' + str(learning_rate[j]) + ': ' + str(training_accuracy))
    print('dev accuracy with lr = ' + str(learning_rate[j]) + ': ' + str(dev_accuracy))





