# deep-q-learner
A Deep-Q Network that navigates Unity's Banana Collector Environment

## 1 Project Details
This environment contains the code files and notebook used to create a neural network that can solve Unity's Banana Collector Environment. The goal of the environment is to have a player, in first person, move around an environment to collect yellow bananas while avoiding blue ones. 

The game awards the player +1 point for every yellow banana collected and -1 point for every blue banana collected. The player can move either forward, backward, left, or right to orient themselves along the space at varying degrees. The game is considered complete when the AI player collected a total of 13 points. 

## 2 How the problem is solved 
This repository uses a Deep-Q learning technique that allos the AI to learn what moves work and what don't based on the environment's feedback, thereby learning to maximise its points over time. There have been modifications to the Deep-Q learner to optmize it. Aside from just using an experience replay buffer, which essentially allows the AI to learn from past experience as well as the immediate state, this model also contains an imlementaion of Dueling Q Networks.

Dueling Q Networks make use of separate decision making strcutures to approximate a given state's value and an approxiamte for the value of it's next action. 

## 3 Getting started with the repository
This repository requires some dependencies to be installed, particulary the Unity Gym. 
Follow these links to install the right dependencies: 
1. Linux: https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip
2. Mac OSX: https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip\
3. Windows 32bit: https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip
4. Windows 64bit: https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip
5. No monitor linux: https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip

## 4 Instructions 

Place the download in this repository in the root folder, and follow along with the Navigation.ipynb Jupyter Notebook. 
