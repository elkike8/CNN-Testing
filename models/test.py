# =============================================================================
# imports
# =============================================================================
from inception_pytorch_v2 import googlenet

import os

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset, DataLoader
from  torch.utils.data.dataset import random_split


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import datetime
import time

# =============================================================================
# dataset
# =============================================================================

torch.manual_seed(1)
start_time_prime = time.time()

print("importing base")

filepath = "C:/Users/FerGo/OneDrive/ACIT/01.2022/Data Mining/Project/Selected X_rays"
list_of_images = os.listdir(filepath)

labels = pd.read_csv("C:/Users/FerGo/OneDrive/ACIT/01.2022/Data Mining/Project/new_labels.csv", index_col=(0))
columns = labels.columns

unique_labels = len(labels.columns)

print("done importing base \nimporting images")


images = []
used_labels = []
a = 0
for files in list_of_images:
    try:
        
        image = plt.imread(filepath + "/" + files)
        image.shape[2]
        print("ignoring image " + str(a))
        a +=1
    except IndexError:
        images.append(image)
        used_labels.append(list_of_images[a])
        a += 1


test_labels = pd.DataFrame(index = used_labels, columns = columns)
for i in used_labels:
    test_labels.loc[used_labels] = labels.loc[used_labels]
    

print("creating dataset") 
test_labels = test_labels.astype(str).astype(int)

    
test_images = np.array(images)
test_images = torch.Tensor(test_images)

test_labels = test_labels.values

x = torch.unsqueeze(test_images, 1)
y = torch.tensor(test_labels).float()

# del(a, columns, filepath, files, i, image, images, labels, list_of_images, 
#    test_images, test_labels, used_labels)

# =============================================================================
# dataset
# =============================================================================

class base(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        
    def __getitem__(self, index):
        return(self.X[index], self.y[index])
    
    def __len__(self):
        return len(self.X)
    

dataset = base(x, y)

train_size = round(len(dataset)*.8) 
test_size = len(dataset) - train_size

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

del(train_size, test_size)

# =============================================================================
# data loader
# =============================================================================

train_loader = DataLoader(train_dataset, batch_size = 10)
test_loader = DataLoader(test_dataset, batch_size = 10)

# =============================================================================
# parameters        
# =============================================================================
print("initializing model")
lr = 0.001
model = googlenet(in_channels = 1, num_classes = unique_labels)
optimizer = optim.Adam(model.parameters(), lr = lr)
loss_fn = nn.BCELoss()
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                 patience=0,
                                                 verbose = True,
                                                 threshold = 0.01)

epochs = 30
training_loss = []
training_accuracy = []
validation_loss = []
validation_accuracy = []


# =============================================================================
# training
# =============================================================================
print("initializing training")

print("the time is " + str(datetime.datetime.now().strftime("%d%b  %H:%M")))

for epoch in range(epochs):
    start_time = time.time()
    model.train()
    correct = 0
    train_batch_loss = []
    
    for x_train, y_train in train_loader:

        y_pred = model(x_train)
        loss = loss_fn(y_pred, y_train)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        train_batch_loss.append(loss.item())
        correct += torch.eq((torch.round(y_pred) == y_train).float().sum(dim=1), 
                            unique_labels).float().sum()
   
    mean_train_loss = np.mean(train_batch_loss)
    training_loss.append(mean_train_loss)
    accuracy = 100 * correct / (len(train_dataset))
    training_accuracy.append(accuracy.item())
    
    model.eval()
    with torch.no_grad():
        t_correct = 0
        test_batch_loss = []
        for x_test, y_test in test_loader:

            yt_pred = model(x_test)
            tloss = loss_fn(yt_pred, y_test)
            test_batch_loss.append(tloss.item())
            t_correct += torch.eq((torch.round(yt_pred) == y_test).float().sum(dim=1), 
                                unique_labels).float().sum()
            
        mean_test_loss = np.mean(test_batch_loss)
        validation_loss.append(mean_test_loss)
        t_accuracy = 100 * t_correct / (len(test_dataset))
        validation_accuracy.append(t_accuracy.item())
    
    scheduler.step(mean_test_loss)
    
    print("[{:02d}] t_loss = {:.3f}\t t_acc = {:.2f}\t v_loss = {:.3f}\t v_acc = {:.2f}". \
          format(epoch+1, training_loss[epoch], accuracy, validation_loss[epoch], t_accuracy))
    
    savepath = "C:/Users/FerGo/OneDrive/ACIT/01.2022/Data Mining/Project/checkpoints/epoch_{}_lr_{}.pt". format(epoch+1, lr)     
    model_state_dict = model.state_dict()
    optimizer_state_dict = optimizer.state_dict()
    
    torch.save({
        "epoch": epoch,
        "model_state_dict": model_state_dict,
        "optimizer_state_dict": optimizer_state_dict},
        savepath)
    training_results = pd.DataFrame()
    training_results["training loss"] = training_loss
    training_results["training accuracy"] = training_accuracy
    training_results["validation loss"] = validation_loss
    training_results["validation accuracy"] = validation_accuracy
    
    training_results.to_csv("C:/Users/FerGo/OneDrive/ACIT/01.2022/Data Mining/Project/checkpoints/results.csv")
    
#    print ("this took "+ str(datetime.timedelta(seconds=round((time.time() - start_time)))) + " to run")
    


# =============================================================================
# plots
# =============================================================================

plt.figure()
plt.plot(training_loss)
plt.plot(validation_loss)
plt.title("model loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(["train", "test"], loc="upper left")
plt.show()
plt.savefig("C:/Users/FerGo/OneDrive/ACIT/01.2022/Data Mining/Project/checkpoints/loss.png")

plt.figure()
plt.plot(training_accuracy)
plt.plot(validation_accuracy)
plt.title("model accuracy")
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.legend(["train", "test"], loc="upper left")
plt.show()
plt.savefig("C:/Users/FerGo/OneDrive/ACIT/01.2022/Data Mining/Project/checkpoints/acc.png")



print ("finalized after running for "+ str(datetime.timedelta(seconds=round((time.time() - start_time_prime)))))
