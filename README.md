# Table of contents
- [Project in the course TDDE19 @ Linköping university](#project-in-the-course-tdde19---link-ping-university)
    + [Problem description](#problem-description)
    + [Aim](#aim)
- [Solution - Xception-CNN model](#solution---Xception---CNN)
    + [Architecture](#architecture)
    + [Dataset](#dataset)
    + [Methodology](#methodology)
- [Running](#running)
    + [Pre-requisites and Installation](#pre-requisites-and-installation)
  * [Description of files](#description-of-files)
    + [Running the files](#running-the-files)

# Project in the course TDDE19 @ Linköping university
### Problem description
The idea behind this topic is to generate caption for a given image. Using this generated caption, select images which are divergent and similar to the input image using text similarity and provide the divergent and similar images as outputs.

### Aim
The aim of this project is to get an understanding regarding the task of image orginality / diversity. Like most tasks in this field of image processing, this task been aided by the ability of deep learning to extract image features. My main goal is to be able to generate a caption for a given input image and retrieve a set of N images farthest from the query image using the text similarity between generated caption and available captions in the system (cloud, database, and etc.). This aproach could be useful for apps like google search, instagram, facebook, and etc. where a user could fetch or view similar images and divergent images based on this approach.

# Solution - Xception-CNN model
### Architecture
In this project, I am going to use the two architectures of Deep Learning and they are CNN and RNN. CNN is usually used for image classification but works very well for image related operations. On the other hand, RNN is usually used to predict the sequence of things such as sentence, number and etc. So combining these two architectures, we are gonna create a system that is going to predict or generate image caption for an input image. The architecture, illustrated in the below figure, is a simplified version based on the CNN-LSTM architecture proposed by Moses Soh in his [article](https://cs224d.stanford.edu/reports/msoh.pdf). Using this simplified model on images for caption generation is a time consuming process. Hence, pre-trained model is applied before the final layer in the architecture to extract the features of given images. These features are saved to a file which can be loaded and used for further process. As a result, training time is significantly reduced.
<div align="center">
<img src="image_Caption_Generator_model_architecture.jpg" width="70%">
</div>

### Dataset
I am going to use the following datasets for training and validation of the model
<ul>
    <li>Flickr_8K => contains images</li>
    <li>Flickr_8K_text => contains a file called flickr8k.token which contains the image and its respective token</li>
</ul>

### Methodology
I am going to use a pre-trained model - <b>Xception</b> for extracting image features and these features are fed into LSTM i.e RNN to generate the captions for the given images. Once the caption for a given image is generated, text similarity i.e. cosine similarity is calculated between the generated caption and set of available captions in the system. If the user requests either the set of similar or divergent images, then those images are retrieved whose similarity scores are above or equal to 90% (similar) and less than 90% (divergent) and displayed to the user based on the number of images that the user had requested.

# Running
### Pre-requisites and Installation
<p> The below given libraries are required to run the project successfully. 
<ul>
    <li> Keras with Tensorflow</li>
    <li> NumPy </li>
    <li> Pickle </li>
    <li> Spacy </li>
    <li> PIL </li>
    <li> String </li>
    <li> OS </li>
    <li> Matplotlib </li>
    <li> NLTK </li>
</ul>
<p> Also, you need to install <b> Jupyter Notebook</b> in order to run the notebooks directly. <p>

## Description of files
**Models/** - This folder contains the finalized deep learning model, deep learning model before and after slight modifications. <br>
**history.p** - This is a pickle file that has the finalized model information such as validation loss and accuracy, training loss and accuracy and etc. which could be used for later purposes. <br>
**Output/** - This folder contains the output of the model, divergent and similar images for a query image, and validation and training loss, validation and training accuracy for the model in the form of images.<br>
**Data/Flicker8k_Dataset/** - This folder contains both the train, test and validation images. <br>
**Data/Flickr8k_text/** - This folder contains the captions along with image names <br>
**Data/Flickr8k_text/Captions.txt** - This file contains all pre-processed captions along with image names. <br>
**Data/Flickr8k_text/tokenized_text.p** - This file is used as the vocabulary to generate captions. <br>
**Models/Finalized_Model/model-ep003-loss3.414-val_loss3.621.h5** - This file is the finalized which has to be loaded using Keras library for prediction.<br>
**and etc.**
### Running the files
In order to see the results, you need to have clone this repository and run the <b>Divergent_Image.ipynb</b> jupyter notebook where you need to change two inputs, i.e., threshold for similarity in <b>text_similarity</b> function and image path in <b>cell 5</b>. Running all the cells in the above mentioned jupyter notebook, you will end up with two kinds of results one with set of divergent images and another with set of similar image for a given query image. If you want to train the model on your machine, you need to run all the cells in <b>Image_Caption_Generation.ipynb</b> jupyter notebook at the end of the runnnig all cells, you would have trained model saved in the models folder.

<b>Note:</b> : Don't change the folder structure or transfer files to other folders, otherwise you won't be able to run the jupyter notebook until you make the changes for input path for files in the jupyter notebooks by yourself.