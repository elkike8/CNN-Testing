from inception_pytorch_v2 import googlenet
import torch
from torch import optim
from torch.utils.data import Dataset, DataLoader

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random


print("importing base")
random.seed(13)

filepath = "C:/Users/FerGo/OneDrive/ACIT/01.2022/Data Mining/Project/Images"
list_of_images_or = os.listdir(filepath)

labels = pd.read_csv("C:/Users/FerGo/OneDrive/ACIT/01.2022/Data Mining/Project/new_labels.csv", index_col=(0))
columns = labels.columns

unique_labels = len(labels.columns)

# =============================================================================
# creating a dataset to predict 
# =============================================================================

print("creating random list")
num_images = 1000

indexes = []
images = 0

while images < num_images:
    a = random.randrange(112120)
    if a in indexes:
        print("ignoring " + str(a))
    else:
        try:
            image = plt.imread(filepath + "/" + list_of_images_or[a])
            image.shape[2]
            print("4d image " + str(a))
        except IndexError:
            indexes.append(a)
            images += 1


list_of_images = [list_of_images_or[x] for x in indexes]


# =============================================================================
# X and Y
# =============================================================================
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


test_loader = DataLoader(dataset, batch_size = 10)

# =============================================================================
# loading the trained model
# =============================================================================

lr = 0.01
model = googlenet(in_channels = 1, num_classes = 15)
optimizer = optim.Adam(model.parameters(), lr = lr)

checkpoint = torch.load("C:/Users/FerGo/OneDrive/ACIT/01.2022/Data Mining/Project/checkpoints/30 lr 0.00005/epoch_19_lr_5e-05.pt")

model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']

model.eval()

# =============================================================================
# Y_pred in batches
# =============================================================================

Y_pred = np.zeros((0, unique_labels))
a = 0
for x_test, y_test in test_loader:
    print(a)
    Y = model(x_test)
    Y = Y.detach().numpy()
    Y_pred = np.vstack((Y_pred, Y))
    a += 1


# =============================================================================
# General ROC
# =============================================================================

y_true = y.numpy()
y_true = y_true.ravel()

y_pred = Y_pred.ravel()



from sklearn.metrics import roc_curve

false_pos, true_pos, thresh = roc_curve(y_true, y_pred)

plt.figure(figsize = (5,5))
plt.plot([0, 1], [0, 1],  linestyle="--")
plt.plot(false_pos, true_pos)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.show()

# =============================================================================
# ROC per label
# =============================================================================
y_true_2 = pd.DataFrame(y.numpy(), columns = columns)
y_pred_2 = pd.DataFrame(Y_pred, columns = columns)

for i in columns:
    print(i)
    false_pos, true_pos, thresh = roc_curve(y_true_2[i], y_pred_2[i])
    globals()["{}". format(i)] = (false_pos, true_pos) 


plt.figure(figsize = (10,10))
plt.plot([0, 1], [0, 1],  linestyle="--")
for i in columns:
    a = globals()["{}". format(i)]   
    plt.plot(a[0], a[1], label = i)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(           loc = "lower center", 
           bbox_to_anchor = (0.5, -0.2), 
           ncol=5)
plt.show()


false_pos, true_pos, thresh = roc_curve(y_true_2[i], y_pred_2[i])
