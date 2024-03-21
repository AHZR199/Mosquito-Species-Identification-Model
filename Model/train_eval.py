# created by abdullah zubair for honours undergraduate thesis (university of calgary 2024)
# part of the soghigian lab (UCVM)  
# linkedin: https://www.linkedin.com/in/a-zubair-calgary/

# this file contains the functions for training and evaluating the mosquito identification model
# it has functions to save and load model checkpoints and find the latest checkpoint
# the train_model function is the main function that trains the model
# it takes the model data loaders loss function optimizer and other hyperparameters as input
# it trains the model for a specified number of epochs and validates it on the validation set
# it keeps track of the training and validation losses accuracies and f1 scores for each epoch
# it saves the best model checkpoint based on the validation loss
# it also implements early stopping to prevent overfitting
# the function returns the trained model and the training and validation metrics

import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torchvision.models import resnet50, ResNet50_Weights
from sklearn.metrics import f1_score

def save_checkpoint(state, checkpoint_dir, learning_rate, batch_size, optimizer_choice, epoch, filename=None):
    if filename is None:
        filename = f"checkpoint_lr_{learning_rate}_bs_{batch_size}_opt_{optimizer_choice}_epoch_{epoch}.pth.tar"
    model_dir = os.path.join(checkpoint_dir, f"lr_{learning_rate}_bs_{batch_size}_opt_{optimizer_choice}")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)  #create model directory if it doesn't exist
    filepath = os.path.join(model_dir, filename)
    torch.save(state, filepath)  #save checkpoint
    print(f"Checkpoint saved to {filepath}")

def load_checkpoint(checkpoint_dir, filename):
    filepath = os.path.join(checkpoint_dir, filename)
    if os.path.isfile(filepath):
        print(f"Loading checkpoint '{filepath}'")
        checkpoint = torch.load(filepath)  #load checkpoint
        return checkpoint
    else:
        print(f"No checkpoint found at '{filepath}', starting from scratch")
        return None

def find_latest_checkpoint(checkpoint_dir, lr, bs, opt):
    pattern = f"checkpoint_lr_{lr}_bs_{bs}_opt_{opt}_epoch_*.pth.tar"
    checkpoints = [f for f in os.listdir(checkpoint_dir) if fnmatch.fnmatch(f, pattern)]  #find checkpoints matching the pattern
    if not checkpoints:
        return None
    checkpoints.sort(key=lambda x: int(x.split('_epoch_')[1].split('.pth.tar')[0]))  #sort checkpoints by epoch
    return checkpoints[-1]  #return the latest checkpoint

def train_model(model, train_loader, val_loader, criterion, optimizer, num_classes, learning_rate, batch_size, optimizer_choice, checkpoint_dir, epochs=150, patience=10, start_epoch=0, best_val_loss=float('inf')):
    patience_counter = 0
    epoch_train_loss, epoch_val_loss, epoch_train_accuracy, epoch_val_accuracy, epoch_f1_scores = [], [], [], [], []

    for epoch in range(start_epoch, epochs):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for inputs, labels in tqdm(train_loader, desc=f'Training Epoch {epoch + 1}'):
            inputs, labels = inputs.cuda(), labels.cuda()  #move inputs and labels to gpu
            optimizer.zero_grad()  #zero gradients
            outputs = model(inputs)  #forward pass
            loss = criterion(outputs, labels)  #calculate loss
            loss.backward()  #backward pass
            optimizer.step()  #update weights
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        train_accuracy = 100 * train_correct / train_total
        epoch_train_loss.append(train_loss / len(train_loader))
        epoch_train_accuracy.append(train_accuracy)

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        all_labels = []
        all_predictions = []
        with torch.no_grad():  #disable gradient calculation during validation
            for inputs, labels in tqdm(val_loader, desc=f'Validation Epoch {epoch + 1}'):
                inputs, labels = inputs.cuda(), labels.cuda()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())

        val_accuracy = 100 * val_correct / val_total
        epoch_val_loss.append(val_loss / len(val_loader))
        epoch_val_accuracy.append(val_accuracy)
        epoch_f1_scores.append(f1_score(all_labels, all_predictions, average='weighted'))

        print(f"Epoch {epoch + 1}: Train Loss: {train_loss / len(train_loader)}, Val Loss: {val_loss / len(val_loader)}, Train Accuracy: {train_accuracy}, Val Accuracy: {val_accuracy}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'train_loss': epoch_train_loss,
                'val_loss': epoch_val_loss,
                'train_accuracy': epoch_train_accuracy,
                'val_accuracy': epoch_val_accuracy,
                'f1_scores': epoch_f1_scores,
                'best_val_loss': best_val_loss,
            }, checkpoint_dir, learning_rate, batch_size, optimizer_choice, epoch)  #save checkpoint
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping due to no improvement")
                break

    return model, epoch_train_loss, epoch_val_loss, epoch_train_accuracy, epoch_val_accuracy, epoch_f1_scores  #return trained model and metrics