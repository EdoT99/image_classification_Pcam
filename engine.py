import torch
import numpy as np
from copy import deepcopy
import time

from model_builder import modelV0
#device agnostic code
device = 'cuda' if torch.cuda.is_available() else 'cpu'


start_time = time.time()

#instantiating the NN before parameters
model = modelV0()
# Defining the model hyperparameters
num_epochs = int
learning_rate = 0.001
weight_decay = 0.01
criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)


def training_session(model: torch.nn.Module,
               train_dataloader: torch.utils.data.DataLoader,
               val_dataloader: torch.utils.data.DataLoader,
               optimizer: torch.optim.Optimizer,
               loss_fn: torch.nn.Module,
               epochs: int,
               accuracy_fn,
               device: torch.device = device
               ):
  '''
  Trains a PyTorch MOdel on train data
  '''
  start_time = time.time()

  train_loss_list = []

  train_acc, loss_train = 0, 0
  model.to(device)
  model.train()

  for batch, (x, y) in enumerate(train_dataloader):
    
    x,y = x.to(device), y.to(device)

    labels = y.float().view(-1, 1).to(device)

    y_logits = model(x)
    loss = loss_fn(y_logits, labels) 
    loss_train += loss

    if batch % 10 == 0:
        print(f'Batch: {batch*len(x)}/{len(train_dataloader)*len(x)} | Loss: {loss}')

    #Note: oncverting logits in the case of binary classification, otehrwise use .argmax()
    predictions = (y_logits > 0.5).float()

    train_acc += accuracy_fn(y_true=labels,y_pred=predictions)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    train_loss_list.append(loss_train)


  loss_train /= len(train_dataloader)
  train_acc /= len(train_dataloader)

  print(f"Train loss: {loss_train} | train accuracy: {train_acc}")
  print('Training time: {:.2f}s'.format(time.time() - start_time))

  '''
  Validates the trained model on validation data
  '''
  val_loss = 0
  val_acc = 0
  model.eval()
  with torch.no_grad():
        for batch, (images, labels) in enumerate(val_dataloader):
            images = images.to(device)
            labels = labels.float().view(-1, 1).to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            #Note: oncverting logits in the case of binary classification, otehrwise use .argmax()

            val_pred = (outputs > 0.5).float()
            val_acc += accuracy_fn(labels,val_pred)

  val_loss /= len(val_dataloader)
  val_acc /= len(val_dataloader)

  best_loss = float('inf')

  # Early stopping
  
  if val_loss < best_loss:
      best_loss = val_loss
      best_model_weights = deepcopy(model.state_dict())  # Deep copy here
      patience = 5  # Reset patience counter
      print("Patience reset")
  else:
      patience -= 1
      if patience == 0:
          print("Early stopping")
          print("Time elapsed: {:.2f}s".format(time.time() - start_time))
          exit()
  
  print(f"Training loss = {loss_train}, Validation loss = {val_loss}")

  
  return train_acc, loss_train, val_acc, val_loss, best_model_weights





def test_session(model: torch.nn.Module,
              test_dataloader: torch.utils.data.DataLoader,
              best_model_weights,
              loss_fn: torch.nn.Module,
              accuracy_fn,
              device: torch.device = device,
              binary: bool = True
              ):
  '''
  Tests and prints a model's loss and accuracy on a testing dataset
  '''
  # Load the best model weights
  if best_model_weights:
      model.load_state_dict(best_model_weights)

  all_labels, all_outputs = [],[]
  model.to(device)
  model.eval()

  with torch.inference_mode():
    test_loss, test_acc = 0,0
    for batch, (x, y) in enumerate(test_dataloader):
      X,Y = x.to(device), y.to(device)

      labels = y.float().view(-1, 1).to(device)
  
      test_pred = model(x)
      test_loss += loss_fn(test_pred, labels)
      #converting logits to predicted labels and compute acc
      #Note: oncverting logits in the case of binary classification, otehrwise use .argmax()
      if binary:
        predictions = (test_pred > 0.5).float()
      else:
         predictions = test_pred.argmax(dim=1)
      test_acc += accuracy_fn(y_true=y,y_pred=predictions)


      #store labels and predictions for later metrics calculation
      all_labels.extend(Y.to(device).numpy())
      all_outputs.extend(predictions.to(device).numpy())
      
    test_loss /= len(test_dataloader)
    test_acc /= len(test_dataloader)
    print(f'Test loss: {test_loss} | Test accuracy: {test_acc}')

    # Convert lists to numpy arrays for metric calculations
    all_labels = np.array(all_labels).flatten()
    all_outputs = np.array(all_outputs).flatten()



  return {'Model': model.__class__.__name__, 'Acc': test_acc ,'Test_loss': test_loss}, all_labels, all_outputs



'''




def evaluate_model(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader,
          val_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device) :     
  
    
    # Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc, best_weights = training_session(model=model,
                                            dataloader1=train_dataloader,
                                            dataloader2=val_dataloader
                                            loss_fn=loss_fn,
                                            optimizer=optimizer,
                                            device=device)
        
        test_loss, test_acc, all_labels, all_predictions = test_session(model=model,
                                            dataloader=test_dataloader,
                                            best_weights = best_weights
                                            loss_fn=loss_fn,
                                            device=device)
       
        # Print out what's happening
        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}"
        )


  
        # Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)
  
    # Return the filled results at the end of the epochs
    return results, np.array(all_labels).flatten(), np.array(all_outputs).flatten()
'''