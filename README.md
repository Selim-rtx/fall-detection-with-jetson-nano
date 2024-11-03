# Human fall detection on video sequences and on livestream video with Nvidia Jetson Nano
* This project aims to detect human falls in order to be warned and help them.
  
## Description

* This project aimes to detect human falls for different use cases such as detection of falls in elderly people living alone or detection of patients falling in their hospital room.
* The project uses an action recognition model that will classify the action into two categories : falling or non-falling. 
* The model has been trained on a dataset produced by Université de Bourgogne (find credits and link below). The dataset has been divided into Falling and Non-falling data.
* The model is based on a combination of two models : a CNN (Resnet-18) and an RNN (LSTM).
* This project is written in Python and uses Pytorch and OpenCV libraries.
* There are three scripts for inference :
  * the first one for inference on a video sequence displaying the prediction rate frame by frame on the command line,
  * the second one for inference on a video sequence and video output with the prediction rate displayed,
  * the third one for inference on a livestream from the camera plugged in the Jetson Nano.

## Deep-dive into the project

For more information about the project, please read my article on my [Medium blog](https://selim-salem.medium.com)

## Getting Started

### Hardware

I used an Nvidia Jetson Nano 4Gb with a 64Gb micro-SD Card on which I allocated 10 Gb for memory swap. To avoid undervoltage and by consequence not having enough power for processing, I used a 5V/4A DC adapter power supply.

### Check your JetPack version
In command line, write the line below to check your JetPack version. In my case, I have JetPack 4.6.1.
```
sudo apt-cache show nvidia-jetpack
```

### Libraries

You will need to install the following libraries:
* PyTorch : Download the correct PyTorch wheel file in [PyTorch for Jetson](https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048) by following the instructions according to your JetPack version. In my case, it was PyTorch v1.10.0 since I have JetPack 4.6.
  * Same for torchvision and even if I use Python3, I had to pip install 'pillow<7'.
* OpenCV although it is normally included in the JetPack installation

### Dataset

In you project folder, create a folder named dataset, in which you will create two other folders, one named falling and the other not_falling, to put in the first one the videos of people falling and the second for people just walking and sitting. 

For this project I used a dataset provided on [Détection de chutes (2014)](https://search-data.ubfc.fr/ub/FR-13002091000019-2024-04-09_Fall-Detection-Dataset.html)

Videos are provided in avi format and you will have to convert them into mp4 format to be process by GStreamer. I cut the videos with ClipChamp to get only the fall for the falling dataset and people walking and sitting for the not_falling dataset. After videos cutting, a video should last around 2 to 4 seconds. This will target specifically the desired action and reduce the workload during processing, especially during training.

### Executing program

Put all the python files and the dataset folder in your project folder. Then run the command lines below. Be sure to put the right absolute path. 

To train the model on your dataset :
```
python3 train_fall_detection.py --dataset /path/to/dataset --epochs 15 --batch_size 1 --sequence_length 40 --output_model trained_fall_lstm_model.pth
```
To run the inference on a video and get only a prediction by frame in the command prompt :
```
python3 run_fall_detection.py --input /path/to/input/video.mp4 --sequence_length 16 --model trained_fall_lstm_model.pth
```
To run the inference on a video and generate the same video with prediction on falls as an overlay :
```
python3 run_fall_detection_video.py --input /path/to/input/video.mp4 --output /path/to/output/prediction_video.mp4 --sequence_length 4 --model trained_fall_lstm_model.pth
```
To run the inference on a live camera stream :
```
python3 run_fall_detection_live.py --camera 0 --sequence_length 16 --model trained_fall_lstm_model.pth
```

## Authors

Selim Salem  
(https://www.linkedin.com/in/selimsalem/)

## Version History

* 1.0
    * Initial Release

## License

This project is licensed under the Apache 2.0 - see the LICENSE.md file for details

## Acknowledgments

Please find below the dataset I used and reference : 
* [Détection de chutes (2014)](https://search-data.ubfc.fr/ub/FR-13002091000019-2024-04-09_Fall-Detection-Dataset.html)

Reference : "I. Charfi, J. Mitéran, J. Dubois, M. Atri, R. Tourki, "Optimised spatio-temporal descriptors for real-time fall detection comparison of SVM and Adaboost based classification", Journal of Electronic Imaging (JEI), Vol.22. Issue.4, pp.17, October 2013. "

My hardware specifications are :
Nvidia Jetson Nano 4 Gb with 64 Gb of ROM (10 Gb of Memory swap)
Webcam : Logitech C270
