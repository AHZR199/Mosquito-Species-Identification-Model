#	created by abdullah zubair for honours undergraduate thesis (university of calgary 2024)
# 		part of the soghigian lab (UCVM)
# 			linkedin: https://www.linkedin.com/in/a-zubair-calgary/

#this file has functions for visualizing and saving different plots and metrics for the mosquito identification model
#it uses matplotliband seaborn libraries to create the plots
#the save_plots function saves the training and validation loss accuracy and f1 score plots
#the plot_and_save_confusion_matrix function plots and saves the confusion matrix
#the confusion matrix shows how well the model performs for each class
#the save_normalized_confusion_matrix function calculates and saves the normalized confusion matrix
#the normalized confusion matrix shows the percentage of correct and incorrect predictions for each class
#the plot_f1_scores_per_class function plots the f1 score for each class
#this helps identify which classes the model performs well on and which ones it struggles with
#the plot_confusion_matrix_by_genus function plots the confusion matrix at the genus level




import os
import matplotlib.pyplot as plt #for creating plots
import seaborn as sns #for creating heatmaps
import numpy as np
from sklearn.metrics import confusion_matrix #for creating confusion matrices

#function to save training and validation plots
def save_plots(train_loss, val_loss, train_accuracy, val_accuracy, f1_scores, model_dir, prefix=""): 
    plt.figure(figsize=(10, 6)) #create figure with size 10x6
    plt.plot(train_loss, label='Training Loss') #plot training loss
    plt.plot(val_loss, label='Validation Loss') #plot validation loss
    plt.title(f'{prefix}Training and Validation Loss') #add title
    plt.xlabel('Epochs') #add x-axis label
    plt.ylabel('Loss') #add y-axis label 
    plt.legend() #add legend
    plt.savefig(os.path.join(model_dir, f"{prefix}loss_plot.png")) #save plot as png
    plt.close() #close plot

    plt.figure(figsize=(10, 6)) #create figure with size 10x6
    plt.plot(train_accuracy, label='Training Accuracy') #plot training accuracy
    plt.plot(val_accuracy, label='Validation Accuracy') #plot validation accuracy
    plt.title(f'{prefix}Training and Validation Accuracy') #add title
    plt.xlabel('Epochs') #add x-axis label
    plt.ylabel('Accuracy') #add y-axis label
    plt.legend() #add legend
    plt.savefig(os.path.join(model_dir, f"{prefix}accuracy_plot.png")) #save plot as png
    plt.close() #close plot

    if f1_scores: #if f1 scores are provided
        plt.figure(figsize=(10, 6)) #create figure with size 10x6
        plt.plot(f1_scores, label='F1 Score') #plot f1 scores
        plt.title(f'{prefix}F1 Scores over Epochs') #add title
        plt.xlabel('Epochs') #add x-axis label
        plt.ylabel('F1 Score') #add y-axis label
        plt.legend() #add legend
        plt.savefig(os.path.join(model_dir, f"{prefix}f1_scores_plot.png")) #save plot as png
        plt.close() #close plot

#function to plot and save confusion matrix
def plot_and_save_confusion_matrix(cm, labels, title, filename):
    plt.figure(figsize=(10, 8)) #create figure with size 10x8
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap='Blues') #create heatmap of confusion matrix
    plt.title(title) #add title
    plt.ylabel('True label') #add y-axis label
    plt.xlabel('Predicted label') #add x-axis label
    plt.savefig(filename) #save plot
    plt.close() #close plot

#function to save normalized confusion matrix
def save_normalized_confusion_matrix(y_true, y_pred, classes, model_dir, prefix=""):
    cm = confusion_matrix(y_true, y_pred) #create confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] #normalize confusion matrix
    plot_and_save_confusion_matrix(cm_normalized, classes, f'{prefix}Normalized Confusion Matrix', os.path.join(model_dir, f"{prefix}normalized_confusion_matrix.png")) #plot and save normalized confusion matrix

#function to plot f1 scores per class
def plot_f1_scores_per_class(f1_scores, classes, model_dir):
    plt.figure(figsize=(12, 6)) #create figure with size 12x6
    plt.bar(range(len(classes)), f1_scores) #create bar plot of f1 scores
    plt.title('F1 Score per Class') #add title
    plt.xlabel('Class') #add x-axis label
    plt.ylabel('F1 Score') #add y-axis label
    plt.xticks(range(len(classes)), classes, rotation=45, ha='right') #add x-tick labels with class names
    plt.tight_layout() #adjust plot layout
    plt.savefig(os.path.join(model_dir, 'f1_score_per_class.png')) #save plot as png
    plt.close() #close plot

#function to plot confusion matrix by genus    
def plot_confusion_matrix_by_genus(y_true, y_pred, classes, model_dir):
    genus_names = sorted(set([name.split('_')[0] for name in classes])) #get unique genus names
    species_to_genus_idx = {idx: genus_names.index(name.split('_')[0]) for idx, name in enumerate(classes)} #create mapping of species index to genus index

    y_true_genus = [species_to_genus_idx[label] for label in y_true] #map true labels to genus labels
    y_pred_genus = [species_to_genus_idx[label] for label in y_pred] #map predicted labels to genus labels

    confusion_mtx_genus = confusion_matrix(y_true_genus, y_pred_genus) #create confusion matrix by genus
    plot_and_save_confusion_matrix(confusion_mtx_genus, genus_names, 'Confusion Matrix by Genus', os.path.join(model_dir, 'confusion_matrix_by_genus.png')) #plot and save confusion matrix by genus

    confusion_mtx_genus_norm = confusion_mtx_genus.astype('float') / confusion_mtx_genus.sum(axis=1)[:, np.newaxis] #normalize confusion matrix by genus
    plot_and_save_confusion_matrix(confusion_mtx_genus_norm, genus_names, 'Normalized Confusion Matrix by Genus', os.path.join(model_dir, 'normalized_confusion_matrix_by_genus.png')) #plot and save normalized confusion matrix by genus