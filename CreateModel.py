import itertools
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from torchvision.models import resnet50, ResNet50_Weights
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

dataset_path = 'ImageBase'
checkpoint_dir = '/work/soghigian_lab/abdullah.zubair/Resnet/'

train_transform = transforms.Compose([
    transforms.Resize((480, 640)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

test_transform = transforms.Compose([
    transforms.Resize((480, 640)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

full_dataset = ImageFolder(root=dataset_path)

def save_checkpoint(state, checkpoint_dir, filename="checkpoint.pth.tar"):
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)
    print(f"Checkpoint saved at '{filepath}'")

def load_checkpoint(checkpoint_dir, filename="checkpoint.pth.tar"):
    filepath = os.path.join(checkpoint_dir, filename)
    if os.path.isfile(filepath):
        print(f"Loading checkpoint '{filepath}'")
        checkpoint = torch.load(filepath)
        return checkpoint
    else:
        print(f"No checkpoint found at '{filepath}' starting from scratch!!!!")
        return None

def train_model(learning_rate, batch_size, optimizer_choice, epochs=100, patience=10, start_epoch=0, best_val_loss=float('inf')):
    train_idx, test_idx = train_test_split(range(len(full_dataset)), test_size=0.2, stratify=full_dataset.targets)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.25, stratify=[full_dataset.targets[i] for i in train_idx])

    train_data = Subset(full_dataset, train_idx)
    val_data = Subset(full_dataset, val_idx)
    test_data = Subset(full_dataset, test_idx)

    train_data.dataset.transform = train_transform
    val_data.dataset.transform = test_transform
    test_data.dataset.transform = test_transform

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)
    test_loader = DataLoader(test_data, batch_size=batch_size)

    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, len(full_dataset.classes))
    model = nn.DataParallel(model).cuda()

    if optimizer_choice == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_choice == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    criterion = nn.CrossEntropyLoss()

    min_val_loss = float('inf')
    patience_counter = 0

    epoch_train_loss = []
    epoch_val_loss = []
    epoch_train_accuracy = []
    epoch_val_accuracy = []
    epoch_f1_scores = []

    for epoch in range(start_epoch, epochs):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for inputs, labels in tqdm(train_loader, desc=f'Training Epoch {epoch+1}'):
            inputs, labels = inputs.cuda(), labels.cuda()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
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
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f'Validation Epoch {epoch+1}'):
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
        epoch_f1_scores.append(f1_score(all_labels, all_predictions, average=None))

        if val_loss < min_val_loss:
            min_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'train_loss': epoch_train_loss,
            'val_loss': epoch_val_loss,
            'train_accuracy': epoch_train_accuracy,
            'val_accuracy': epoch_val_accuracy,
            'f1_scores': epoch_f1_scores,
            'best_val_loss': min_val_loss,
        }, checkpoint_dir, filename=f"checkpoint_lr_{learning_rate}_bs_{batch_size}_opt_{optimizer_choice}_epoch_{epoch+1}.pth.tar")

    return model, train_loader, test_loader, epoch_train_loss, epoch_val_loss, epoch_train_accuracy, epoch_val_accuracy, epoch_f1_scores

learning_rates = [0.001, 0.0001, 0.0015, 0.002, 0.0025]
batch_sizes = [16, 32]
optimizers = ['Adam', 'SGD']

best_loss = float('inf')
best_lr = None
best_bs = None
best_opt = None
best_model = None
best_train_loss = None
best_val_loss = None
best_train_accuracy = None
best_val_accuracy = None
best_f1_scores = None

for lr, bs, opt in itertools.product(learning_rates, batch_sizes, optimizers):
    print(f"Training with lr={lr}, batch_size={bs}, optimizer={opt}")
    checkpoint_filename = f"checkpoint_lr_{lr}_bs_{bs}_opt_{opt}.pth.tar"
    checkpoint = load_checkpoint(checkpoint_dir, filename=checkpoint_filename)

    if checkpoint is not None:
        start_epoch = checkpoint['epoch']
        best_val_loss = checkpoint['best_val_loss']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
    else:
        start_epoch = 0
        best_val_loss = float('inf')

    try:
        model, train_loader, test_loader, train_loss, val_loss, train_accuracy, val_accuracy, f1_scores = train_model(lr, bs, opt, start_epoch=start_epoch, best_val_loss=best_val_loss)
        min_current_val_loss = min(val_loss)
        if min_current_val_loss < best_loss:
            best_loss = min_current_val_loss
            best_lr = lr
            best_bs = bs
            best_opt = opt
            best_model = model
            best_train_loss = train_loss
            best_val_loss = val_loss
            best_train_accuracy = train_accuracy
            best_val_accuracy = val_accuracy
            best_f1_scores = f1_scores
    except RuntimeError as e:
        print(f"Failed with lr={lr}, batch_size={bs}, optimizer={opt} due to RuntimeError: {e}")

best_model.eval()
y_true = []
y_pred = []
with torch.no_grad():
    for inputs, labels in tqdm(test_loader, desc='Testing'):
        inputs, labels = inputs.cuda(), labels.cuda()
        outputs = best_model(inputs)
        _, predicted = torch.max(outputs, 1)
        y_true.extend(labels.tolist())
        y_pred.extend(predicted.tolist())

test_report = classification_report(y_true, y_pred, target_names=full_dataset.classes, output_dict=True)
print(classification_report(y_true, y_pred, target_names=full_dataset.classes))

torch.save(best_model.state_dict(), f'{checkpoint_dir}THE_ONE.pth')

print("Best Model Training Loss per Epoch:", best_train_loss)
print("Best Model Validation Loss per Epoch:", best_val_loss)
print("Best Model Training Accuracy per Epoch:", best_train_accuracy)
print("Best Model Validation Accuracy per Epoch:", best_val_accuracy)
print("Best Model F1 Scores per Epoch:", best_f1_scores)

plt.figure(figsize=(10, 6))
plt.plot(best_train_loss, label='Training Loss')
plt.plot(best_val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig(f'{checkpoint_dir}loss_plot.svg')

plt.figure(figsize=(10, 6))
plt.plot(best_train_accuracy, label='Training Accuracy')
plt.plot(best_val_accuracy, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig(f'{checkpoint_dir}accuracy_plot.svg')

plt.figure(figsize=(10, 6))
plt.bar(full_dataset.classes, np.mean(best_f1_scores, axis=0))
plt.title('F1 Score per Class')
plt.xlabel('Species')
plt.ylabel('F1 Score')
plt.xticks(rotation=90)
plt.savefig(f'{checkpoint_dir}f1_score_plot.svg')

confusion_mtx = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(12, 10))
sns.heatmap(confusion_mtx, annot=True, fmt='g', xticklabels=full_dataset.classes, yticklabels=full_dataset.classes)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig(f'{checkpoint_dir}confusion_matrix.svg')

species_list = full_dataset.classes
print("The model is trained to identify the following species:")
for species in species_list:
    print(species)
