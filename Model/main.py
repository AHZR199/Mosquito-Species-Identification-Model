# created by abdullah zubair for honours undergraduate thesis (university of calgary 2024) 
# part of the soghigian lab (UCVM)
# linkedin: https://www.linkedin.com/in/a-zubair-calgary/

#this is the main file that puts everything together for training and evaluating the mosquito identification model


#it uses the functions from the data_utils train_eval and visualize files
#it defines the dataset path checkpoint directory and hyperparameters to search over
#it trains the model for each combination of hyperparameters and saves the best model and its metrics
#it loads the dataset using the load_dataset function from data_utils
#it trains the model using the train_model function from train_eval
#it saves the training and validation metrics using the save_plots function from visualize
#after training it evaluates the best model on the test set and prints the classification report
#it generates various visualizations using functions from visualize stuff
#it also then prints the best model parameters and metrics and saves the best model checkpoint so that it cna be used for the flask web app 


import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet50, ResNet50_Weights
from sklearn.metrics import classification_report, f1_score
from tqdm import tqdm  
import itertools
import matplotlib.pyplot as plt

from data_utils import load_dataset
from train_eval import train_model, load_checkpoint, find_latest_checkpoint
from visualize import save_plots, plot_f1_scores_per_class, plot_confusion_matrix_by_genus, save_normalized_confusion_matrix

dataset_path = 'Images' #CHANGE BASED ON WHERE IMAGE DATASEST IS LOCATED
checkpoint_dir = '/work/soghigian_lab/abdullah.zubair/test' #also change based on where to save checkpoints

learning_rates = [0.001, 0.0001, 0.01]
batch_sizes = [16, 32]
optimizers = ['Adam', 'SGD']

best_loss = float('inf') #This is done so that it doesnt error out bc Python cannot compare None with a float using <
best_lr = None
best_bs = None  
best_opt = None
best_model = None
best_train_loss = None
best_val_loss = None
best_train_accuracy = None
best_val_accuracy = None
best_f1_scores = None

for lr, bs, opt in itertools.product(learning_rates, batch_sizes, optimizers):  #iterate over hyperparameter combinations
    model_dir = os.path.join(checkpoint_dir, f"lr_{lr}_bs_{bs}_opt_{opt}")
    os.makedirs(model_dir, exist_ok=True)  #create model directory if it doesn't exist
    
    print(f"Training with lr={lr}, batch_size={bs}, optimizer={opt}")
    checkpoint_filename = find_latest_checkpoint(checkpoint_dir, lr, bs, opt)  #find latest checkpoint
    if checkpoint_filename:
        checkpoint = load_checkpoint(checkpoint_dir, checkpoint_filename)  #load checkpoint
        if checkpoint is not None:
            start_epoch = checkpoint['epoch']
            best_val_loss = checkpoint['best_val_loss']
            model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            num_features = model.fc.in_features
            model.fc = nn.Linear(num_features, len(full_dataset.classes))
            model = nn.DataParallel(model).cuda()  #use data parallelism for multi-gpu training
            optimizer = optim.Adam(model.parameters(), lr=lr) if opt == 'Adam' else optim.SGD(model.parameters(), lr=lr, momentum=0.9)
            model.load_state_dict(checkpoint['state_dict'])  #load model state dict from checkpoint
            optimizer.load_state_dict(checkpoint['optimizer'])  #load optimizer state dict from checkpoint
        else:
            start_epoch = 0
            best_val_loss = float('inf')
    else:
        start_epoch = 0
        best_val_loss = float('inf')

    full_dataset, train_loader, val_loader, test_loader = load_dataset(dataset_path, checkpoint_dir, bs)  #load dataset
    num_classes = len(full_dataset.classes)
    criterion = nn.CrossEntropyLoss()  #define loss function
    
    model, train_loss, val_loss, train_accuracy, val_accuracy, f1_scores = train_model(
        model, train_loader, val_loader, criterion, optimizer, num_classes, lr, bs, opt, checkpoint_dir,
        start_epoch=start_epoch, best_val_loss=best_val_loss)  #train model

    if val_loss[-1] < best_loss:  #update best model if current model has lower validation loss
        best_loss = val_loss[-1]
        best_lr = lr
        best_bs = bs
        best_opt = opt
        best_model = model
        best_train_loss = train_loss
        best_val_loss = val_loss
        best_train_accuracy = train_accuracy
        best_val_accuracy = val_accuracy
        best_f1_scores = f1_scores

    save_plots(train_loss, val_loss, train_accuracy, val_accuracy, f1_scores, model_dir, prefix="")  #save plots

# Evaluate the best model on the test set
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

print(classification_report(y_true, y_pred, target_names=full_dataset.classes))  #print classification report
plot_f1_scores_per_class(f1_score(y_true, y_pred, average=None), full_dataset.classes, checkpoint_dir)  #plot f1 scores per class
plot_confusion_matrix_by_genus(y_true, y_pred, full_dataset.classes, checkpoint_dir)  #plot confusion matrix by genus
save_normalized_confusion_matrix(y_true, y_pred, full_dataset.classes, checkpoint_dir)  #save normalized confusion matrix

print(f"Best Model Parameters: Learning Rate = {best_lr}, Batch Size = {best_bs}, Optimizer = {best_opt}")
print("Best Model Training Loss per Epoch:", best_train_loss)
print("Best Model Validation Loss per Epoch:", best_val_loss) 
print("Best Model Training Accuracy per Epoch:", best_train_accuracy)
print("Best Model Validation Accuracy per Epoch:", best_val_accuracy)
print("Best Model F1 Scores per Epoch:", best_f1_scores)

# Save loss and accuracy plots for the best model
save_plots(best_train_loss, best_val_loss, best_train_accuracy, best_val_accuracy, best_f1_scores, checkpoint_dir, prefix="best_")  #save plots for best model

torch.save(best_model.state_dict(), f'{checkpoint_dir}/BEST_MODEL.pth')  #save best model state dict