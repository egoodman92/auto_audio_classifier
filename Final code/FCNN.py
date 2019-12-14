#Final version Dec 13 2019
import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from sklearn import preprocessing
import numpy as np
from sklearn.metrics import confusion_matrix


n_class = 7

# read the csv file
X_train_path = 'car_data_X_balanced_7.csv'
Y_train_path = 'car_data_Y_balanced_7.csv'
X_train = pd.read_csv(X_train_path, header= None)
Y_train = pd.read_csv(Y_train_path, header= None)

#UNCOMMENT THIS CODE FOR THE SELECT ARRAY OF FEATURES
#X_train = np.asarray(X_train)
#X_train = np.column_stack((X_train[:,515], X_train[:,318], X_train[:,518], X_train[:,45], X_train[:,35], X_train[:,515], X_train[:,32], X_train[:,43],  X_train[:,406], X_train[:,588], X_train[:,664], X_train[:,349], X_train[:,660], X_train[:,554], X_train[:,648], X_train[:,310], X_train[:,615], X_train[:,523], X_train[:,62], X_train[:,269], X_train[:,666], X_train[:,514], X_train[:,261], X_train[:,253], X_train[:,228], X_train[:,518], X_train[:,54], X_train[:,58], X_train[:,646], X_train[:,123], X_train[:,588], X_train[:,350], X_train[:,646], X_train[:,78], X_train[:,19], X_train[:,672], X_train[:,129], X_train[:,0], X_train[:,1], X_train[:,2]))
#X_train = pd.DataFrame(X_train)

n_examples = X_train.values[2:].shape[0]
print("n_examples  = " , n_examples )
Y_train_one_hot = np.zeros((n_examples, n_class))

# shuffle all data
randIndexTotal = np.random.permutation(n_examples)
X_train_values = X_train.values[2:][randIndexTotal]
Y_train_values = Y_train.values[2:][randIndexTotal]

for index, item in enumerate(Y_train_values):
    Y_train_one_hot[index, int(item[0])] = 1

#Split all data into test and train data
X_train = torch.tensor(X_train_values[0:324]).float()
Y_train_class = torch.tensor(Y_train_one_hot[0:324], dtype=torch.long)
Y_train = torch.max(Y_train_class, 1)[1]
X_test = torch.tensor(X_train_values[324:]).float()
Y_test_class = torch.tensor(Y_train_one_hot[324:], dtype=torch.long)
Y_test = torch.max(Y_test_class, 1)[1]

# data normalization
X_train_norm = preprocessing.scale(X_train, axis=0, with_mean=True, with_std=True, copy=True)
X_train_norm = torch.tensor(X_train_norm).float()
X_test_norm = preprocessing.scale(X_test, axis=0, with_mean=True, with_std=True, copy=True)
X_test_norm = torch.tensor(X_test_norm).float()

# Define evaluation matric
def calculate_accuracy(X, Y):
    outputs = model(X)
    Y_pred = torch.max(outputs, 1)[1]
    print("Y_pred = ", Y_pred)
    Y_pred = Y_pred.float()
    num_correct = torch.sum(Y.float() == Y_pred)
    acc = (num_correct * 100.0 / len(Y)).item()
    C = confusion_matrix(Y, Y_pred)
    print((C)/C.astype(np.float).sum(axis=0))
    return acc


learning_rate =  [.1, .03, .01, 0.003, .001] #

train_acc = {}

n_iteration = 30000
train_loss_for_all_rate = []


for j in range(len(learning_rate)):
    print("learning_rate = ", j)
    model = nn.Sequential(
        nn.Linear(678, 20),
        nn.ReLU(),
        nn.Linear(20, n_class),
        nn.Softmax())
    # Define the cost function
    criterion = nn.CrossEntropyLoss()
    # Define the optimizer, learning rate
    optimizer = optim.Adam(model.parameters(), lr=learning_rate[j])
    # Record cost per interation
    cost_per_iteration = []
    # dev_cost_per_iteration = []

    if learning_rate[j] not in (train_acc.keys()):
        train_acc[learning_rate[j]] = []
        # dev_acc[learning_rate[j]] = []
        # test_acc[learning_rate[j]] = []

    # Train the network on the training data
    training_loss = []
    for i in range(n_iteration):
        # if (i%100 == 0):
        #     print("iter = ", i)

        # zero the parameter gradients
        optimizer.zero_grad()
        # forward propogation
        outputs = model(X_train_norm)

        # calculate the loss
        loss = criterion(outputs, Y_train)

        # backpropogation + update parameters
        loss.backward()
        optimizer.step()

        cost = loss.item()
        cost_per_iteration.append(cost)

    train_loss_for_all_rate.append(cost_per_iteration)
    # plt.plot(cost_per_iteration)
    # plt.show()

    # Calculate accuracy
    training_accuracy = calculate_accuracy(X_train_norm, Y_train)
    # dev_accuracy = calculate_accuracy(X_dev_norm, Y_dev_class)

    print('training accuracy with lr =  ' + str(learning_rate[j]) + ': ' + str(training_accuracy))

    test_accuracy = calculate_accuracy(X_test_norm, Y_test)
    # print('dev accuracy with lr = ' + str(learning_rate[j]) + ': ' + str(dev_accuracy))
    print('test accuracy with lr = ' + str(learning_rate[j]) + ': ' + str(test_accuracy))
