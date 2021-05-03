# Cassava_Pytorch
## Snapmix in PyTorch for Kaggle Cassava Comp 2021

Code for [Kaggle Competition] (https://www.kaggle.com/c/cassava-leaf-disease-classification/overview)

# These research-papers were used for this code:

* SnapMix: Semantically Proportional Mixing for Augmenting Fine-grained Data - http://arxiv.org/abs/2012.04846

And to understand the SnapMix article:
* Mixup: Beyond Empirical Risk Minimization - http://arxiv.org/abs/1710.09412
* Learning Deep Features for Discriminative Localization - http://arxiv.org/abs/1512.04150

This network uses a [pretrained Resnet50 implementation](https://www.kaggle.com/pytorch/resnet50) was trained for 20 epochs, submitted and trained for another 20 epochs while in submission. 

## Parameters used:
* Images where resized to 256 x 256 and center-cropped to 224 x 224.
* Training/ Evaluation data: 0.8/0.2 of the supplied dataset - public dataset approx. 21.000 images.
* Snapmix augmentation was applied half of the time.
* The classifier of the Resnet50 was replaced by a 5 classes classifier and pretrained for 3 epochs. 
* After the 3rd epoch the last convolutional layer of the Resnet50 was trained, too.
* Optimizer: 
	*Stochastic Gradiend Descent
	*learning-rate = 0.001
	* momentum = 0.9
	* no lr-scheduler
* Batch-size = 32

## Results:
* Private Score: 0.7679
* Public Score: 0.7682

# Future improvements:
* include bias-weights in algorithm
* implement learning-rate-scheduler
* use/ implement weighted loss-function: dataset is heavily unbalanced
* use cross-validation / more epochs to train
* use "intermediate"-attention/ localization like in the "Learning Deep Features for Discriminative Localization" - paper
* try normal-distribution instead of beta for the size of the random boxes
* determine confusion-matrix to better see where the network fails



