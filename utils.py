
"""
Contains various utility functions for PyTorch model training and saving.
"""
import torch
from pathlib import Path
import matplotlib as plt

def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):
  """Saves a PyTorch model to a target directory.

  Args:
    model: A target PyTorch model to save.
    target_dir: A directory for saving the model to.
    model_name: A filename for the saved model. Should include
      either ".pth" or ".pt" as the file extension.

  Example usage:
    save_model(model=model_0,
               target_dir="models",
               model_name="05_going_modular_tingvgg_model.pth")
  """
  # Create target directory
  target_dir_path = Path(target_dir)
  target_dir_path.mkdir(parents=True,
                        exist_ok=True)

  # Create model save path
  assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
  model_save_path = target_dir_path / model_name

  # Save the model state_dict()
  print(f"[INFO] Saving model to: {model_save_path}")
  torch.save(obj=model.state_dict(),
             f=model_save_path)
  


def accuracy_fn(y_true,y_pred):
    
    correct = torch.eq(y_true,y_pred).sum().item()
    acc = (correct/len(y_pred))*100
  
    return acc
  

def image_draw(image,label,class_names):
    plt.figure(figsize=(6, 6))
    if len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1):
        cmap = 'gray'
        image = image.squeeze()  # Remove singleton dimensions if necessary
    else:
        cmap = None
    plt.imshow(image.squeeze(),cmap=cmap)
    plt.colorbar()
    plt.grid(False)
    plt.title(f'Label: {class_names[label]}')
    plt.show()

def plot_losses(loss_values_train,loss_values_test,epochs_count):

  plt.figure(figsize=(16, 8))
  # Plot training and test loss curves
  plt.plot(epochs_count, loss_values_train, label="train loss")
  plt.plot(epochs_count, loss_values_test, label="test loss")

  plt.title('Training and Test Loss Curves')
  plt.ylabel('Loss')
  plt.xlabel('Epochs')
  plt.legend()
  plt.show()
