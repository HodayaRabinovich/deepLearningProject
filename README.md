# Bird Species Classification - Transfer Learning & Networks Robustness

In this project we used tranfer learning to adjust pre-trained network to a new dataset.
We "attacked" the networks with noise, checked their robustness and tried to improve one of the network.

## Dataset
we used [325 bird species](https://www.kaggle.com/gpiosenka/100-bird-species) from kaggle

* 224x224x3 color images
* 325 classes
* 47,382 trainig images, 1,625 test images, 1,625 validation images

## Transfer Learning
we chose 4 pre-trained models to work with - ResNet50, ResNet18, vgg16 and DenseNet.
Applying feature extraction we traind only the last FC layer in each network.
we got the following results:

|               | ResNet50  | ResNet18  | Vgg16    | DenseNet  |
| ------------- |:---------:|:---------:|:--------:|----------:|
| Test accuracy | 85.47%    | 89.1%     | 89.84%   | 93.04%    |

## Noise & Robustness
we wanted to check the network results for images that were taken at different times of the day (night, sunset, etc.).
So we added Color Jitter to the test set. In addition we added gaussian noise ~ N(0,1) which doesn't have a visual influence.
The results:
|                     | ResNet50  | ResNet18  | Vgg16    | DenseNet  |
| -------------       |:---------:|:---------:|:--------:|----------:|
| Noisy test accuracy | 52.8%    | 53.41%     | 57.12%   | 66.33%    |

## Augmentation
From now on we focused on ResNet18.
In attempt to improve the results we added augmentation and trained the last FC layer again.

|                     | ResNet18  | ResNet18 with aug |
| -------------       |:---------:| -----------------:|
| Test accuracy       | 89.1%     | 78.8%             | 
| Noisy test accuracy | 53.41%    | 76.8%             | 

this result doesn't satisfy so we decided to fine tune the model
now we trained all the layers with low learning rate.
result:
|                     | ResNet18  | ResNet18 with aug | ResNet18 Fine Tuning & aug |
| -------------       |:---------:| -----------------:| --------------------------:|
| Test accuracy       | 89.1%     | 78.8%             | 96.27                      |
| Noisy test accuracy | 53.41%    | 76.8%             | 95.4%                      |


## Conclusion
* Feature extraction works well to classify ordinary dataset
* However, when adding noise it didn't achieve satisfying result, so we used fine tunning (which takes more time)
 
 
 ## Files in this repository
| File Name       |                         Desciption                              |
| -------------   |:--------------------------------------------------------------: | 
| main.py         | build all the model, save them and test (original & noisy test) | 
| func.py         | all the functions to run main                                   | 
| images.py       | demonstate datasets images with and without augmentation        | 

