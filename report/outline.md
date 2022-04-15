
---
# Introduction and Background Research

## Introduction  
What is the project trying to solve and why,  Explained at a high level.

## Literature Review 
### A Short History of Computer Vision
Some tools and approaches that will be referenced later.  Chronological order preferred.
### Computer Vision for Chess  
Existing solutions broken down into two subset.  Each solution explained at varying 
levels of detail with a healthy amount of criticism and praise for varying approaches.
### Prior Work From the Author
Work I have done previously on robotics, vision and machine learning.

---
# Methods

## Data Collection
Why data is important and some of aims that my approach has in comparison to previous works.
### Sensors
The sensors setup I used and why.
### Auto-Labelling
How I went about collecting labelled data and why this method was chosen.  The amount of time it saved and where the data was stored.
### Dataset Versioning
Where the data was stored and how it was managed.  The advantages and dis-advantages of this and perhaps any changes I would make.

## Model Architecture
Given this data what is are model meant to do.
### Experiment Tracking
How I went about this from a project management point of view
### Board Segmentation
What I used for board segmentation, expressing that this isn't the focus of the project but needed to get to Piece Recognition.
### Piece Recognition
Some of the models I tested - why those models and some of the problems I encountered and how I solved them.
Inludes all features of the model like (optimizer, layers, loss function, multitask heads, transfer learning)
### Augmentation
After a good model was found, how data augmentation can help and why I picked the augmentations I did.

## Inference
The goal of inference.  How human interaction can help, but the problems with that.
### The Not-So-Reliable Approach
The flow of data and different data structures.  Some added headings like (motion detection) to remove noise.  Memory to stabilise the result and using game knowledge (i.e. BoardState vs VisionState) to record a full PGN game.

---
# Results

## Model Evaluation
Overview of the techniques I used to evaluate my model throughout. Express that every change was evaluated.
### Board Segmentation
Won't linger here.  Briefly mention the speed and limitations.  
### Multitask Learning
The most important addition so will compare the accuracy and latency of multitask learning vs not.
### Piece Recognition
Created Confusion Matrix and Top-losses visualizations and how it helped me better understand my data.
### Hyper-parameters
Show the tensorflow coordinates view to view how hyper-parameter were chosen.
### Deep Dive into CNNs
Explain some insights gained from looking at filter visualizations.

## Realtime Analysis
An evaluation of my solution when trying to generate chess PGN files of full games.
### Trials
Some trials, visualized in a table to show the performance of my solution, perhaps with different lighting.

## Comparison to Existing Solutions
Overall, what did my solution do differently and what did it do the same.
### Quantitative Comparison
Explain how the limited data made it difficult to compare quantitatively, but here was much setup.
Then the results.

---
# Discussion

## Conclusion
What I achieved.
### Piece Recognition
### Dataset Management

## Ideas for future work
What I'd do next.