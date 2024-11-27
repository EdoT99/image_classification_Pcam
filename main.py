import os
import time
from copy import deepcopy
from datetime import datetime
import torch 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from model_builder import modelV0
from engine import training_session, test_session
from utils import accuracy_fn
import h5py 

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score, ConfusionMatrixDisplay, roc_curve, \
    classification_report
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset



results_training = pd.DataFrame(columns = ['epoch','Train_size','Acc','loss_train','loss_val'])
results_test = pd.DataFrame(columns = ['epoch','Acc','AUC(ROC)','F1-score'])

# Initializing normalizing transform for the dataset
# scaling pixels in the range 0,0 : 1.0 
#
normalize_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

# Downloading the PCam dataset into train and test sets
# if you already ahve the datasets downloaded, leave as it is
# otherwise change download = True


full_train_dataset = torchvision.datasets.PCAM(
    root="~/project/my_project/data/train", transform=normalize_transform,
    split='train', download=False
)



full_test_dataset = torchvision.datasets.PCAM(
    root="~/project/my_project/data/test", transform=normalize_transform,
    split='test', download=False
)


#train_subset_size = 25000
train_subset_size = 6400
test_subset_size = 1920

# Create indices and subsets for train and test datasets
train_indices, _ = train_test_split(range(len(full_train_dataset)), train_size=train_subset_size, random_state=42)
test_indices, _ = train_test_split(range(len(full_test_dataset)), train_size=test_subset_size, random_state=42)


batch_size = 128
#splitting train dataset into train set and validation set
train_indices, val_indices = train_test_split(train_indices, test_size=0.2, random_state=42)

#subsetting training, validation and testing into smaller fraction
train_dataset = Subset(full_train_dataset, train_indices)
val_dataset = Subset(full_train_dataset, val_indices)
test_dataset = Subset(full_test_dataset, test_indices)

print('-----------------------------------------------')
print(f'Train dataset size: {len(train_dataset)} images')
print(f'Validation set size: {len(val_dataset)} images')
print(f'Test set size {len(test_dataset)} images')

#preparing loading 

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# set device agnostics

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = modelV0()

# setting parameters
num_epochs = 10
learning_rate = 0.001
weight_decay = 0.01

# set loss function 
criterion = torch.nn.BCELoss()
# set backpropagation algorithm
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)


start_time = time.time()


#training and validating the model
for epoch in range(num_epochs):
    print(f'{int(epoch+1)} epoch')
    print('-----------------------------------------------')

    train_acc, loss_train, val_acc, val_loss, best_weights = training_session(model,train_loader,val_loader,optimizer,criterion,num_epochs,accuracy_fn)
    
    row_result = {'epochs':epoch,'Train_size':len(train_dataset),'Acc':train_acc,'loss_train':loss_train,'loss_val':val_loss}
    #results_training.loc[len(results_training)] = row_result

#testing tuned model on test set
stats, all_true_labels, all_predicted_labels, = test_session(model,test_loader,best_weights,criterion,accuracy_fn)


#computing AUC(ROC) metric and F1-score
auc_roc = roc_auc_score(all_true_labels, all_predicted_labels)
f1 = f1_score(all_true_labels, all_predicted_labels)

row_test = {'Epochs':num_epochs,'Acc': stats['Acc'],'AUC(ROC)':auc_roc,'F1-score':f1}

print(f"AUC (Trained): {auc_roc:.4f}")
print(f"F1 Score: {f1:.4f}")

fpr_trained, tpr_trained, _ = roc_curve(all_true_labels, all_predicted_labels)


save_folder = '../my_project/results'

endTime = datetime.now()            
result_date = endTime.strftime("%Y%m%d_%H%M")

'''
full_path = f'{save_folder}/{result_date}_training_validation.csv'
results_training.to_csv(full_path, index=False, sep=';', decimal=',')







# Now let's calculate the AUC and ROC curve for the untrained model
untrained_model = modelV0().to(device)  # Initialize a new model without loading weights
untrained_outputs = []

test_session(untrained_model,test_loader,best_model_weights=None,)
untrained_model.eval()
with torch.no_grad():
    for i, (images, _) in enumerate(test_loader):
        images = images.to(device)
        outputs = untrained_model(images)
        untrained_outputs.extend(outputs.cpu().numpy())

# Convert the untrained model's outputs to a numpy array
untrained_outputs = np.array(untrained_outputs).flatten()

# Calculate the AUC and ROC curve for the untrained model
auc_untrained = roc_auc_score(all_labels, untrained_outputs)
fpr_untrained, tpr_untrained, _ = roc_curve(all_labels, untrained_outputs)
print(f"AUC (Untrained): {auc_untrained:.4f}")
'''
# Plotting the ROC curve for both trained and untrained models on the same graph
plt.figure()
plt.plot(fpr_trained, tpr_trained, label=f"Trained Model AUC = {auc_roc:.4f}")
#plt.plot(fpr_untrained, tpr_untrained, linestyle='--', label=f"Untrained Model AUC = {auc_untrained:.4f}")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison: Trained vs Untrained Model")
plt.legend()
plt.show()


print("Time elapsed: {:.2f}s".format(time.time() - start_time))













