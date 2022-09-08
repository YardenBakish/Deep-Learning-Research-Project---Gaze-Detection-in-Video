# Deep-Learning-Workshop - In Proggress - anticipated date: 14.9.22

# Deep-Learning-Workshop - Depth-Based Gaze Target Detection in Video
Deep Learning Workshop Github repository

## About
The repository contains the code and documentation for Deep-Learning Workshop

## Demo Colab Link:
https://colab.research.google.com/drive/14ua6sTa-IgaPJZPLxqAvN3LhQx2SmI1J#scrollTo=G6PLExr7G728

## Demo
To run a demo Please click on the provided colab link above. Simply connect to a GPU runtime and run the (only) cell.
You will notice in your file directory that a new directory called 'output'  has been created. this directory contains a visualization for gaze detection for all of the frames provied at 'data_demo' directory.

**Input Arguments**

The demo is executed via the following Linux command:
```bash
python run_demo.py --person left
```
There are two people appearing in the frames. You have the option to choose either 'left' or 'right' for gaze detection visualzation for the left person or the right person respectively. 



