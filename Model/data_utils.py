# created by abdullah zubair for honours undergraduate thesis (university of calgary 2024)
# part of the soghigian lab (UCVM)
# linkedin: https://www.linkedin.com/in/a-zubair-calgary/


# this file has functions to load the dataset apply transformations and split the data into train validation and test sets
# the train_transform and test_transform define the data augmentation and preprocessing steps for the images
# these transformations help make the model better and improve its performance
# the save_dataset_split_info function saves the dataset split information like file paths species counts and genera counts
# the load_dataset function is the main function that loads the dataset using pytorch's imagefolder
# it splits the dataset into train validation and test subsets using scikit-learns train_test_split
# it applie the transformations to the data subsets and creates data loaders for efficient data loading during training and evaluation

# basically  this file takes care of all the datarelated tasks needed for the mosquito identification model

import os
import torch
import torchvision.transforms as transforms #for data augmentation and preprocessing
from torchvision.datasets import ImageFolder #for loading image data
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split #for splitting data into train, val, test

#define data augmentation and preprocessing for training images
train_transform = transforms.Compose([
    transforms.Resize((224, 224)), #resize images to 224x224
    transforms.RandomHorizontalFlip(), #randomly flip images horizontally
    transforms.RandomVerticalFlip(), #randomly flip images vertically
    transforms.RandomRotation(15), #randomly rotate images by up to 15 degrees
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1), #randomly change brightness, contrast, saturation, and hue
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5), #randomly apply perspective transformation
    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=5), #randomly apply affine transformations
    transforms.RandomGrayscale(p=0.1), #randomly convert images to grayscale
    transforms.GaussianBlur(kernel_size=3), #apply gaussian blur to images
    transforms.ToTensor(), #convert images to tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), #normalize images
])

#define preprocessing for testing and validation images
test_transform = transforms.Compose([
    transforms.Resize((224, 224)), #resize images to 224x224
    transforms.ToTensor(), #convert images to tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), #normalize images
])

#function to save dataset split information
def save_dataset_split_info(dataset, indices, split_name, checkpoint_dir): 
    species_count = {} #dictionary to store species counts
    genera_count = {} #dictionary to store genus counts
    with open(f'{checkpoint_dir}/{split_name}_split_info.txt', 'w') as file: #open file for writing
        for idx in indices: #iterate over indices
            class_name = dataset.classes[dataset.targets[idx]] #get class name
            genus_name = class_name.split('_')[0] #get genus name
            species_count[class_name] = species_count.get(class_name, 0) + 1 #update species count
            genera_count[genus_name] = genera_count.get(genus_name, 0) + 1 #update genus count
            file_path = dataset.imgs[idx][0] #get file path
            file.write(f"{file_path}\n") #write file path to file

        file.write("\nSpecies Count:\n") #write species counts to file
        for species, count in species_count.items():
            file.write(f"{species}: {count}\n") 

        file.write("\nGenera Count:\n") #write genus counts to file
        for genus, count in genera_count.items():
            file.write(f"{genus}: {count}\n")

#function to load dataset
def load_dataset(dataset_path, checkpoint_dir, batch_size):
    full_dataset = ImageFolder(root=dataset_path) #load full dataset

    #split dataset into train and test
    train_idx, test_idx = train_test_split(range(len(full_dataset)), test_size=0.15, stratify=full_dataset.targets) 
    #split train into train and val
    train_idx, val_idx = train_test_split(train_idx, test_size=0.1765, stratify=[full_dataset.targets[i] for i in train_idx])

    #save dataset split information
    save_dataset_split_info(full_dataset, train_idx, "train", checkpoint_dir) 
    save_dataset_split_info(full_dataset, val_idx, "val", checkpoint_dir)
    save_dataset_split_info(full_dataset, test_idx, "test", checkpoint_dir)

    #create subsets for train, val, and test
    train_data = Subset(full_dataset, train_idx) 
    val_data = Subset(full_dataset, val_idx)
    test_data = Subset(full_dataset, test_idx)

    #apply transforms to subsets
    train_data.dataset.transform = train_transform
    val_data.dataset.transform = test_transform
    test_data.dataset.transform = test_transform

    #create data loaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)
    test_loader = DataLoader(test_data, batch_size=batch_size)

    return full_dataset, train_loader, val_loader, test_loader #return full dataset and data loaders