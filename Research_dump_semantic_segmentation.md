# Semantic Segmentation 
## Introduction

The aim of this Computer Vision task is to assign every pixel of the image to a class.

This is an example of semantic image segmentation
![alt_text](https://www.jeremyjordan.me/content/images/2018/05/Screen-Shot-2018-05-17-at-7.42.16-PM.png)

The way we traet standard categorical values, here target will also be one-hot encoded by creating an output channel for all possible classes.

For visulization ![alt_text](https://www.jeremyjordan.me/content/images/2018/05/Screen-Shot-2018-05-16-at-9.36.00-PM.png)
![alt_text](https://www.jeremyjordan.me/content/images/2018/05/Screen-Shot-2018-05-16-at-9.36.38-PM.png)

## Loss Functions

The loss functions which are used are cross-entropy loss and soft dice loss. We are well aware of the cross-entropy loss in multi-classfication tasks. Soft dice loss is the extension of the dice coefficents developed for binary data.

![alt_text](./dice.png)

Implementation of soft dice loss
<script src="https://gist.github.com/jeremyjordan/9ea3032a32909f71dd2ab35fe3bacc08.js"></script>

## Evaluation Metrics

Mean IoU is general evaluation metric used in semantic image segmentation challenges.

![alt_text](./IoU.png)

IoU score is calculated for each class with which global and mean IoU score is calculated.

This measure can also be represnted in terms of True positives, False Positives and False Negatives.

![alt_text](./IoU_2.png)

