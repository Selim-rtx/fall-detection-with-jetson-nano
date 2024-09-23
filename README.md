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

## Illustration

[![Watch the video](https://img.youtube.com/vi/OjoA3c8PRKA/0.jpg)](https://www.youtube.com/watch?v=OjoA3c8PRKA)

## Getting Started

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
Référence : "I. Charfi, J. Mitéran, J. Dubois, M. Atri, R. Tourki, "Optimised spatio-temporal descriptors for real-time fall detection comparison of SVM and Adaboost based classification", Journal of Electronic Imaging (JEI), Vol.22. Issue.4, pp.17, October 2013. "
Videos are provided in avi format and you will have to convert them into mp4 format to be process by GStreamer. 

### Installing

I coded my project in a virtual environement in Visual Studio Code:
```
python3 -m venv venv
```
```
venv\Scripts\activate
```
I installed all the libraries above but for llama I used the following : 
```
python -m pip install llama-cpp-python==0.2.26 --prefer-binary --extra-index-url=https://jllllll.github.io/llama-cpp-python-cuBLAS-wheels/AVX2/cu122
```
It's a llama-cpp wheel that I found on this github : [llama-cpp-python cuBLAS wheels](https://github.com/jllllll/llama-cpp-python-cuBLAS-wheels)
I chose this version because I had issues with Wheels installation and I wanted to be sure that I was using CUDA for inference, and I found exactly what I wanted thanks to this github page. As you will see, you have to know if your CPU uses AVX, and chose the right version according to your python and CUDA Toolkit version you have on your computer.

You can download the json files and use them locally with JSONLoader.

## Datasets and model choice
Concerning the datasets I used, to make the vector store, I chose these links to a github proposing medical Q&As in json format: 
* [eHealthforumsQAs](https://github.com/LasseRegin/medical-question-answer-data/blob/master/ehealthforumQAs.json)
* [icliniqQAs](https://github.com/LasseRegin/medical-question-answer-data/blob/master/icliniqQAs.json)
And I generated with ChatGPT 4o, a json with symptoms and the medical department where to go. You will find the file named hospital_departement.json in this github.

For the model, I tried quantized Mistral-7b and Llama-13b from Hugging Face :
* [Mistral-7b](bhttps://huggingface.co/TheBloke/Mistral-7B-OpenOrca-GGUF)
* [Llama-13b](https://huggingface.co/TheBloke/Llama-2-13B-chat-GGUF)

I recommend to fine tune this model on your dataset before using it for inference. I followed this tutorial to do so :
[Fine tuning](https://rentry.org/cpu-lora#appendix-a-hardware-requirements)

## GPU offloading
When you configure your model, you can choose through gpu_layers, how many layers of the model you can offload to the GPU. I offload half (22/44) because when I offload them all, the GPU was too overloaded probably because it's too short in RAM. I tested only 1 layer on the GPU and the rest on the CPU, and it was slower. The inference took 4 minutes in my case because I used a laptop configuration (RTX 2060M). For production, a desktop with an RTX 30 or 40 series with at least 12 Gb of VRAM or a professional GPU such as Quadro RTX would be more appropriate and way faster. 

### Executing program

```
python3 doc_chatbot.py
```
You will find at the end of the output messages in the CLI, an URL that will display the chatbot in your main browser.

## Help

To verify your CUDA version, you can execute this prompt in CLI.
```
nvidia-smi
```

## Authors

Selim Salem  
(https://www.linkedin.com/in/selimsalem/)

## Version History

* 0.1
    * Initial Release

## License

This project is licensed under the MIT License - see the LICENSE.md file for details

## Acknowledgments

Please find below documentations I used to write my code : 
* [Nvidia Embeddings](https://nvidia.github.io/GenerativeAIExamples/latest/notebooks/10_RAG_for_HTML_docs_with_Langchain_NVIDIA_AI_Endpoints.html))
* [LangChain QA](https://python.langchain.com/v0.2/docs/tutorials/local_rag/)

My hardware specifications are :
CPU : Intel Core i7-1165G7 (Quad-Core 1.2 GHz - 2.8 GHz / 4.7 GHz Turbo - 8 Threads - Cache 12 Mo - TDP 28 W) 
RAM : 16 Go DDR4 3200 Mhz
GPU : Nvidia RTX 2060 M (6 Go V-RAM)
