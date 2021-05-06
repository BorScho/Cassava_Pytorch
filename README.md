# Cassava_Pytorch
## Snapmix in PyTorch for Kaggle Cassava Comp 2021

Code for [Kaggle Competition] (https://www.kaggle.com/c/cassava-leaf-disease-classification/overview)

# These research-papers were used for this code:

* SnapMix: Semantically Proportional Mixing for Augmenting Fine-grained Data - http://arxiv.org/abs/2012.04846

And to understand the SnapMix article:
* Explanation of the CAM (class attention map): Learning Deep Features for Discriminative Localization - http://arxiv.org/abs/1512.04150
* What is the rational behind mixing image-data : Mixup: Beyond Empirical Risk Minimization - http://arxiv.org/abs/1710.09412

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
* implement learning-rate-scheduler
* use/ implement weighted loss-function: dataset is unbalanced
* use cross-validation / more epochs to train
* use "intermediate"-attention/ localization like in the "Learning Deep Features for Discriminative Localization" - paper
* try normal-distribution instead of beta for the size of the random boxes: I don't know why beta anyway...
* determine confusion-matrix to better see where the pediction fail


## Explanation of the Snapmix Algorithm

Snapmix ("Semantically Proportional Mixing") is an augmentation technique for fine-grained image data. 
Fine-grained data are data, where subtle differences decide on the classification.

Two other mixing algorithms Mixup and Cutmix showed that augmenting images by somehow "mixing" them for training, can improve the accuracy of a network for classification.
Mixup considers pixel-wize convex combinations of two images, i.e. construct a pixel p in a new synthetic image by setting 
<img src="https://render.githubusercontent.com/render/math?math=%5Cbegin%7Balign*%7D%0Ap%20%3D%20(1-%20%5Clambda)%20*%20p1%20%2B%20%20%5Clambda%20*%20p2%5C%5C%0A%5Cend%7Balign*%7D">
for every pixel p1 of image1 and p2 of image2. The same transformation is applied to the labels, say: l1, l2, of the images, thus the synthetic label for the constructed image is: 
<img src="https://render.githubusercontent.com/render/math?math=%5Cbegin%7Balign*%7D%0Al%20%3D%20(1-%20%5Clambda)%20*%20l1%20%2B%20%20%5Clambda%20*%20l2%5C%5C%0A%5Cend%7Balign*%7D">

Cutmix cuts out some part/box from both images and uses one cutout box as a patch for the other. That seems to work similar to drop-out for neurons - the network has to learn varying features for classification, since some of them are opaqued some of the time. The label of the synthetic image is again a convex combination, where lambda is the ratio of the area of the cutout box to the area of the whole image: 
* (1- $\lambda$) for the label of the image patched into, because this image is loosing some of it's original content 
* $\lambda$ for the label of the image from which the patch is taken, because some of it's content is added to the synthetic picture.

The drawback of cutting out and pasting a patch from one image into another might be, that the randomly generated patch does not contain any valuable information about the label (like e.g. only contains parts of the background) and when patched into the other picture might even cover the area where almost all classifying information of this picture was located - if we for example have a picture with a cat on a lawn and another with a dog on a lawn, we might after patching end up with a picture of a lawn - no cat, no dog. Nevertheless the image will have a synthetic label indicating some percentage of a dog and some percentage of a cat in the picture.

### Reducing label-noise by snapmix
To mittigate this introduction of label-noise into the training-data, snapmix uses the class activation map (CAM) of every image to determine the percentage of classifying-content of each pixel, i.e. how much each pixel of the image contributes to the classification of the image according to it's label. The "Learning Deep Features..." paper explains the construction of CAMs.
By knowing the amount of classifying-content in each pixel, called the "semantic percentage" in the snapmix paper, we can then calculate the amount present in the patch taken from image2, say $\rho_{2}$, and the amount present in the area of image1, that will be covered by the patch, say $\rho_{1}$. The synthetic label, l, of the patch-work image will be: l = (1 - $\rho_{1}$) * l1 + $\rho_{2}$ * l2

### Extension of the loss function by linearity
How to use the synthetic labels? What does it help to know, that a picture contains 40% cat and 60% dog (the rare "doggish catdog")? We use labels to calculate the loss, and we can do so by extending the loss function to synthetic labels by linearity:

Let the synthetic label be: l = (1 - $\rho_{1}$) * l1 + $\rho_{2}$ * l2 and let $y^hat$ be the predicted class for the synthetic image. We then define:

loss(l, $y^hat$) = loss((1 - $\rho_{1}$) * l1 + $\rho_{2}$ * l2, $y^hat$) := (1 - $\rho_{1}$) * loss(l1, $y^hat$) + $\rho_{2}$ * loss(l2, $y^hat$)

i.e. the loss-function is extended to synthetic labels by linearity. 
This requires to have l1 and l2 one-hot encoded such that l is a linear combination of two linearly independend vectors - otherwize the above would not be a well defined construction. For implementation one-hot encoding is not necessary, we just calculate the two losses on the right side of the equation.

Another feature introduced in the snapmix paper is asymmetry of the boxes: the boxes are not taken from the same places in the two images, neither do they need to have the same size (as is both the case in the cutmix algorithm). As is shown in the snapmix-paper, asymetry contributes to further increase accuracy.

### Construction of the Class Activation Map (CAM)
The class activation map necessary for snapmix is explained in the "Learning Deep Features..." paper mentioned above. It uses the last CNN-feature map of a back-bone network just before average-pooling and the fully-connected classifier. In my implementation the backbone used is an imageNet pretrained ResNet50.

Let:
F($I_{i}$) - feature map of the last conv-layer of the back-bone for the i-th image $I_{i}$, i.e. F($I_{i}$)
$y_{i}$ - the label for the i-th image $I_{i}$



### The Semantic Percentage Map (SPM)




