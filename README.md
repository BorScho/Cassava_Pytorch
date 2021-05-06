# Cassava_Pytorch
## Snapmix in PyTorch for Kaggle Cassava 2021 Competition

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
<!-- $$
p = (1 - \lambda) * p1 + \lambda * p2
$$ --> 

<div align="center"><img style="background: white;" src="https://render.githubusercontent.com/render/math?math=p%20%3D%20(1%20-%20%5Clambda)%20*%20p1%20%2B%20%5Clambda%20*%20p2"></div>

for every pixel p1 of image1 and p2 of image2.\\
The same transformation is applied to the labels, say: l1, l2, of the images, thus the synthetic label for the constructed image is: 
<!-- $$
l = (1 - \lambda) * l1 + \lambda * l2
$$ --> 

<div align="center"><img style="background: white;" src="https://render.githubusercontent.com/render/math?math=l%20%3D%20(1%20-%20%5Clambda)%20*%20l1%20%2B%20%5Clambda%20*%20l2"></div>

Cutmix cuts out some part/box from both images and uses one cutout box as a patch for the other. That seems to work similar to drop-out for neurons - the network has to learn varying features for classification, since some of them are opaqued some of the time. The label of the synthetic image is again a convex combination, where lambda is the ratio of the area of the cutout box to the area of the whole image. We use 
<!-- $$
(1- \lambda)
$$ --> 

<div align="left"><img style="background: white;" src="https://render.githubusercontent.com/render/math?math=(1-%20%5Clambda)"></div> 

for the image to be patched into, because this image is loosing some of it's original content, and 
<!-- $$
\lambda
$$ --> 

<div align="left"><img style="background: white;" src="https://render.githubusercontent.com/render/math?math=%5Clambda"></div>

for the label of the image from which the patch is taken, because some of it's content is added to the synthetic picture.

The drawback of cutting out and pasting a patch from one image into another might be, that the randomly generated patch does not contain any valuable information about the label (like e.g. only contains parts of the background) and when patched into the other picture might even cover the area where almost all classifying information of this picture was located - if we for example have a picture with a cat on a lawn and another with a dog on a lawn, we might after patching end up with a picture of a lawn - no cat, no dog. Nevertheless the image will have a synthetic label indicating some percentage of a dog and some percentage of a cat in the picture.


### Reducing label-noise by snapmix
To mittigate this introduction of label-noise into the training-data, snapmix uses the class activation map (CAM) of every image to determine the percentage of classifying-content of each pixel, i.e. how much each pixel of the image contributes to the classification of the image according to it's label. The "Learning Deep Features..." paper explains the construction of CAMs.
By knowing the amount of classifying-content in each pixel, called the "semantic percentage" in the snapmix paper, we can then calculate the amount present in the patch taken from image2, say \rho_{2}, and the amount present in the area of image1, that will be covered by the patch, say \rho_{1}. The synthetic label, l, of the patch-work image will be: 
<!-- $$
l = (1 - \rho_{1}) * l1 + \rho_{2} * l2
$$ --> 

<div align="center"><img style="background: white;" src="https://render.githubusercontent.com/render/math?math=l%20%3D%20(1%20-%20%5Crho_%7B1%7D)%20*%20l1%20%2B%20%5Crho_%7B2%7D%20*%20l2"></div>


### Extension of the loss function by linearity
How to use the synthetic labels? What does it help to know, that a picture contains 40% cat and 60% dog (the rare "doggish catdog")? We use labels to calculate the loss, and we can do so by extending the loss function to synthetic labels by linearity. Let

<!-- $$
\hat y 
$$ --> 

<div align="center"><img style="background: white;" src="https://render.githubusercontent.com/render/math?math=%5Chat%20y%20"></div> 

be the predicted class for the synthetic image. We then define:

<!-- $$
loss(l, \hat y) = loss((1 - \rho_{1}) * l1 + \rho_{2} * l2, \hat y) := (1 - \rho_{1}) * loss(l1, \hat y) + \rho_{2} * loss(l2, \hat y) 
$$ --> 

<div align="center"><img style="background: white;" src="https://render.githubusercontent.com/render/math?math=loss(l%2C%20%5Chat%20y)%20%3D%20loss((1%20-%20%5Crho_%7B1%7D)%20*%20l1%20%2B%20%5Crho_%7B2%7D%20*%20l2%2C%20%5Chat%20y)%20%3A%3D%20(1%20-%20%5Crho_%7B1%7D)%20*%20loss(l1%2C%20%5Chat%20y)%20%2B%20%5Crho_%7B2%7D%20*%20loss(l2%2C%20%5Chat%20y)%20"></div>

i.e. the loss-function is extended to synthetic labels by linearity. 
This requires to have l1 and l2 one-hot encoded such that l is a linear combination of two linearly independend vectors - otherwize the above would not be a well defined construction. For implementation one-hot encoding is not necessary, we just calculate the two losses on the right side of the equation.

Another feature introduced in the snapmix paper is asymmetry of the boxes: the boxes are not taken from the same places in the two images, neither do they need to have the same size (as is both the case in the cutmix algorithm). As is shown in the snapmix-paper, asymetry contributes to further increase accuracy.


### Construction of the Class Activation Map (CAM)
The class activation map necessary for snapmix is explained in the "Learning Deep Features..." paper mentioned above. It uses the last CNN-feature map of a back-bone network just before average-pooling and the fully-connected classifier. In my implementation the backbone used is an imageNet pretrained ResNet50.

We use the following notation:
<!-- $$
F(I_{i}) \text{- feature map of the last conv-layer of the back-bone for the i-th image} I_{i} 
$$ --> 

<div align="center"><img style="background: white;" src="https://render.githubusercontent.com/render/math?math=F(I_%7Bi%7D)%20%5Ctext%7B-%20feature%20map%20of%20the%20last%20conv-layer%20of%20the%20back-bone%20for%20the%20i-th%20image%7D%20I_%7Bi%7D%20"></div>

<!-- $$
y_{i} \text{- the label for the i-th image} I_{i}
$$ --> 

<div align="center"><img style="background: white;" src="https://render.githubusercontent.com/render/math?math=y_%7Bi%7D%20%5Ctext%7B-%20the%20label%20for%20the%20i-th%20image%7D%20I_%7Bi%7D"></div>

The channels of the feature-map F are average pooled and feed into a fully connected layer with 5 neurons and softmax for output.
Let the weight-matrix for this fc-layer be W and the bias vector b. Then for the i-th image I_{i} the input L to the class k will be:
<!-- $$
L(i,k) = b_{k} + \sum_{j} W_{k,j} \sum_{x,y} F_{j}(I_{i})(x,y)
$$ --> 

<div align="center"><img style="background: white;" src="https://render.githubusercontent.com/render/math?math=L(i%2Ck)%20%3D%20b_%7Bk%7D%20%2B%20%5Csum_%7Bj%7D%20W_%7Bk%2Cj%7D%20%5Csum_%7Bx%2Cy%7D%20F_%7Bj%7D(I_%7Bi%7D)(x%2Cy)"></div>

where:
<!-- $$
F_{j}(I_{i})(x,y) \test{ is the average over the j-th channel of the feature-map for the image.}
$$ -->

<div align="center"><img style="background: white;" src="https://render.githubusercontent.com/render/math?math=F_%7Bj%7D(I_%7Bi%7D)(x%2Cy)%20%5Ctest%7B%20is%20the%20average%20over%20the%20j-th%20channel%20of%20the%20feature-map%20for%20the%20image.%7D"></div> 

We can interchange the summations in the formula, because all channels in the feature-map have the same size:
<!-- $$
L(i,k) = b_{k} + \sum_{x,y} \sum_{j} W_{k,j} F_{j}(I_{i})(x,y)
$$ --> 

<div align="center"><img style="background: white;" src="https://render.githubusercontent.com/render/math?math=L(i%2Ck)%20%3D%20b_%7Bk%7D%20%2B%20%5Csum_%7Bx%2Cy%7D%20%5Csum_%7Bj%7D%20W_%7Bk%2Cj%7D%20F_%7Bj%7D(I_%7Bi%7D)(x%2Cy)"></div>

The contribution to the classification of the image as belonging to it's class y_{i} is:
<!-- $$
L(i,y_{i}) = b_{y_{i}} + \sum_{x,y} M(I_{i})(x,y)
$$ --> 

<div align="center"><img style="background: white;" src="https://render.githubusercontent.com/render/math?math=L(i%2Cy%7Bi%7D)%20%3D%20b_%7By_%7Bi%7D%7D%20%2B%20%5Csum_%7Bx%2Cy%7D%20M(I_%7Bi%7D)(x%2Cy)"></div>

where we defined M as:
<!-- $$
M(I_{i})(x,y) := \sum_{j} W_{y_{i},j} F_{j}(I_{i})(x,y)
$$ --> 

<div align="center"><img style="background: white;" src="https://render.githubusercontent.com/render/math?math=M(I_%7Bi%7D)(x%2Cy)%20%3A%3D%20%5Csum_%7Bj%7D%20W_%7By_%7Bi%7D%2Cj%7D%20F_%7Bj%7D(I_%7Bi%7D)(x%2Cy)"></div>

**being the contribution of the pixel at (x,y) to the classification of the image as of class y_{i}**


### The Semantic Percentage Map (SPM)
To get the relative percentage of the contribution of each pixel we normalize M to sum to one:
<!-- $$
S(I_{i})(x,y) = \frac{M(I_{i})(x,y)}{\sum_{x,y} M(I_{i})(x,y)}
$$ --> 

<div align="center"><img style="background: white;" src="https://render.githubusercontent.com/render/math?math=S(I_%7Bi%7D)(x%2Cy)%20%3D%20%5Cfrac%7BM(I_%7Bi%7D)(x%2Cy)%7D%7B%5Csum_%7Bx%2Cy%7D%20M(I_%7Bi%7D)(x%2Cy)%7D"></div>

*We have to be cautious here*: the coordinates (x,y) designate a pixel in the feature-map, not the original image I! Since we want to cut and patch on the original image, we have to upsample our feature-map. Let this upsampling be done by a function \Phi . We will then call our function S from above the "Semantic Percentage Map (SPM):
<!-- $$
SPM(I_{i})(x,y) = \frac{\Phi(M(I_{i}))(x,y)}{\sum_{x,y} \Phi(M(I_{i}))(x,y)}
$$ --> 

<div align="center"><img style="background: white;" src="https://render.githubusercontent.com/render/math?math=SPM(I_%7Bi%7D)(x%2Cy)%20%3D%20%5Cfrac%7B%5CPhi(M(I_%7Bi%7D))(x%2Cy)%7D%7B%5Csum_%7Bx%2Cy%7D%20%5CPhi(M(I_%7Bi%7D))(x%2Cy)%7D"></div>

where x,y now are pixel-coordinates on the original image I_{i}


### Snapmixing
To apply snapmix we calculate the SPMs of image1 and image2, cut box1 from Image1 and box2 from Image2 and calculate their semantic content to finally get our \rho factors for the synthetic label, l:
<!-- $$
\rho_{1} = 1 - \sum_{x,y \in box1} SPM(I_{1})(x,y) 
$$ --> 

<div align="center"><img style="background: white;" src="https://render.githubusercontent.com/render/math?math=%5Crho_%7B1%7D%20%3D%201%20-%20%5Csum_%7Bx%2Cy%20%5Cin%20box1%7D%20SPM(I_%7B1%7D)(x%2Cy)%20"></div>

and 
<!-- $$
\rho_{2} = \sum_{x,y \in box2} SPM(I_{2})(x,y)
$$ --> 

<div align="center"><img style="background: white;" src="https://render.githubusercontent.com/render/math?math=%5Crho_%7B2%7D%20%3D%20%5Csum_%7Bx%2Cy%20%5Cin%20box2%7D%20SPM(I_%7B2%7D)(x%2Cy)"></div>

and thus our synthetic label is:
<!-- $$
l = \rho_{1} * y_{1} + \rho_{2} * y_{2}
$$ --> 

<div align="center"><img style="background: white;" src="https://render.githubusercontent.com/render/math?math=l%20%3D%20%5Crho_%7B1%7D%20*%20y_%7B1%7D%20%2B%20%5Crho_%7B2%7D%20*%20y_%7B2%7D"></div> 

The only thing left to do, is to resize the cut out box2 to the size of box1, paste it into image1 and supply it to the training-loop, using the loss function as given above.




