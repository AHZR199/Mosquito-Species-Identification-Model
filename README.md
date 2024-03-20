# Mosquito-Species-Identification-Model

# Overview
Created by [Abdullah Zubair](https://www.linkedin.com/in/a-zubair-calgary/) as part of Honours Undergraduate Thesis [(University of Calgary BHSc)](https://cumming.ucalgary.ca/bhsc) titled 'Development and Evaluation of a Machine Learning-Based Image Classification Web Application for Mosquito Species Identification for Species Present in Alberta, Canada'

## Main Objectives for this Project
i.	Build a comprehensive image dataset by capturing images of mosquito species currently found within Alberta and combining with published image datasets.

ii.	Create and evaluate a CNN model using transfer learning with a known and tested architecture and train, validate, and test the model using the image dataset. 

iii.	Develop a user-friendly Flask web application that allows users to easily upload multiple images and receive species identification results from the trained CNN model.

# Details
## Folders and Files
### Folders
* App - This folder contains the files needed to run the Flask app along with the .pth file which has the weights for the model used by the app
* Model - This folder contains all the python files used to create, train, and test the model along with creating figures based on the model metrics
* Images - This folder is the image datset created for this project and as such it contains all the images captured with each subfolder using the convention 'genus_species'
### Files
* App/environment.yml - contains a list of dependencies and their versions and is to be used to create the conda environment to run the Flask app
* App/app.py - This is main Python file that contains the Flask app
* App/index.html - Home page/entry point for a web application
* App/result.html - Page that displays the output from the ML model for each image
* App/best_model.pth - Stores the PyTorch model which is used for the Flask app
* Model/data_utils.py -
* Model/visualize.py -
* Model/train_eval.py - 
* Model/main.py -

# Installation
## Conda Environment
The environment.yml file contains a list of dependencies and their versions. First install Conda if you havent then to set up a virtual environment using conda use the command

conda env create -f environment.yml -n (env_name). 

Then, activate the environment by running 

conda activate (env_name)

## To Run the Files to Create the Model
Change the directories used to save the checkpoints and obtain the images within the main.py file (lines 32 and 33)

# Running the Web App

The Flask app has only been tested on a HPC specifically the ARC HPC at the University of Calgary, however, it can run on a local machine with a GPU.

## On Unix Systems
1. Open Terminal, ssh and Login to ARC, and Run an interactive session on one of ARC's GPU nodes
	
Here is the command for that:

salloc --ntasks=1 --cpus-per-task=4 --mem=16G --time=00:10:00 --gpus-per-node=1

You can adjust the time argument depending on how long you plan to run the app (currently set for 10 minutes)
	
2. Activate conda environment made earlier

3. Navigate to the app directory and run the app

Once running it should give you text that shows you that the application is running and accessible on a certain IP addresse
Here is an example:

Running on http://127.0.0.1:8888
			
4. Open another terminal/command prompt window and connect to ARC using the normal SSH client with the tunnel option
		
Here are the commands for that:

ssh (Your Username)@arc.ucalgary.ca -L 8888:(specific node you are on):8888

This command depends on which GPU node you were assigned in your interactive session
Here is an example of that command:

ssh john.doe@arc.ucalgary.ca -L 8888:fg1:8888

5. Now go back to the first terminal/command prompt window with the app running and cope the IP address and paste it into your browser
	
The IP address that starts with 127 is the one that should work and then pasting that address into Chrome or Safari should
let you then access and use the app!
