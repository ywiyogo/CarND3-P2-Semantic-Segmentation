# Semantic Segmentation in Advanced Deep Learning

#Project Summary

In this project, I've built a Fully Convolutional Network in order to segment camera images, whether an area is a road or not. The camera is mounted inside a vehicle. The approach of the semantic segmentation is based on the this [publication](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf).

After running this project, the main program will generate a `logs` directory for the visualization using tensorboard and a `runs` directory which contains the segmented images from the input datasets.

## Network Architecture
Tensorflow provides a tool `tensorboard` to visualize the Convolutional Network. The below figure shows the overview of the VGG16 architecture and the additional upsampling and skipping layers in this project:

![architecture][image1]

## Result

Based on my limited experiments, I found out that the number of the epochs returns more visual difference on the images. Applying these fix hyperparameters:

* `learning rate`: 0.001
* `keep_prob`: 0.5

the bellow tables shows the difference of the result on some images.

| Epoch 10 | Epoch 40 |
|:--:|:--:|
|![Epoch 10][image10-1]|![Epoch 40][image40-1]|
|![Epoch 10][image10-2]|![Epoch 40][image40-2]|
|![Epoch 10][image10-3]|![Epoch 40][image40-3]|
|![Epoch 10][image10-4]|![Epoch 40][image40-4]|

As shown in the above table, the higher the number of epoch, the better is the segmentation result. The cross entropy loss reaches 0.02553. To support this observation I add the cross entropy loss to the tensorflow summary. The graph of the cross entropy loss can be visualized as this below figure:

![loss graph][image2]


---
### Udacity's Project Introduction
In this project, you'll label the pixels of a road in images using a Fully Convolutional Network (FCN).

### Setup
##### Frameworks and Packages
Make sure you have the following is installed:
 - [Python 3](https://www.python.org/)
 - [TensorFlow](https://www.tensorflow.org/)
 - [NumPy](http://www.numpy.org/)
 - [SciPy](https://www.scipy.org/)
##### Dataset
Download the [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) from [here](http://www.cvlibs.net/download.php?file=data_road.zip).  Extract the dataset in the `data` folder.  This will create the folder `data_road` with all the training a test images.

### Start
##### Implement
Implement the code in the `main.py` module indicated by the "TODO" comments.
The comments indicated with "OPTIONAL" tag are not required to complete.
##### Run
Run the following command to run the project:
```
python main.py
```
**Note** If running this in Jupyter Notebook system messages, such as those regarding test status, may appear in the terminal rather than the notebook.

### Submission
1. Ensure you've passed all the unit tests.
2. Ensure you pass all points on [the rubric](https://review.udacity.com/#!/rubrics/989/view).
3. Submit the following in a zip file.
 - `helper.py`
 - `main.py`
 - `project_tests.py`
 - Newest inference images from `runs` folder  (**all images from the most recent run**)
 
## How to write a README
A well written README file can enhance your project and portfolio.  Develop your abilities to create professional README files by completing [this free course](https://www.udacity.com/course/writing-readmes--ud777).

[//]: # (Image References)
[image1]: ./imgs/architecture_semantic.png
[image2]: ./imgs/cross_entropy.png
[image10-1]: ./imgs/ep10_um_000003.png
[image10-2]: ./imgs/ep10_um_000004.png
[image10-3]: ./imgs/ep10_um_000010.png
[image10-4]: ./imgs/ep10_um_000011.png
[image40-1]: ./imgs/ep40_um_000003.png
[image40-2]: ./imgs/ep40_um_000004.png
[image40-3]: ./imgs/ep40_um_000010.png
[image40-4]: ./imgs/ep40_um_000011.png