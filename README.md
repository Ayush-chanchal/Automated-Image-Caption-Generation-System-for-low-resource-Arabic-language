
# Automated Image Caption Generation System for Low-Resource Arabic Language

This repository contains the code for an **Automated Image Caption Generation System** designed for the **low-resource Arabic language**. The system uses **VGG16** for visual feature extraction and **LSTM networks** for sequence generation, producing high-quality captions in Arabic for given images. It addresses the challenge of generating coherent image captions in Arabic, a language with limited NLP resources, by combining computer vision and natural language processing.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Project Architecture](#project-architecture)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Sample Results](#sample-results)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

The **Automated Image Caption Generation System** is a deep learning-based project aimed at generating accurate captions in Arabic for images. The model integrates **VGG16** (a convolutional neural network) for feature extraction and **LSTM** (a recurrent neural network) to generate language-based captions. This project is intended to bridge the gap in low-resource language applications by providing accurate descriptions for images, which can be useful in **assistive technology**, **content creation**, and more.

## Features

- **Arabic Captioning**: Automatically generates captions in Arabic, addressing the challenges of limited resources for Arabic NLP.
- **High Accuracy**: Extensive testing and validation ensure coherent, accurate captions.
- **Flexible and Scalable**: Can be adapted to other low-resource languages or extended for larger datasets.
- **Assistive Applications**: Useful for assistive technologies, content creation, and image-based applications.

## Project Architecture

1. **Image Feature Extraction**: The **VGG16** model is used to extract essential features from input images.
2. **Sequence Generation**: The extracted features are processed by an **LSTM network** to generate descriptive sentences in Arabic.
3. **Captioning Pipeline**: The system tokenizes and preprocesses captions, allowing the model to understand the image context and generate relevant descriptions.

The pipeline is designed to effectively capture and interpret image context, producing meaningful captions in Arabic.

## Dataset

We use the **Flickr30k** dataset, a comprehensive image dataset containing thousands of images with captions. For this project:
- **Arabic Translations**: We preprocessed and translated the captions into Arabic, making the dataset suitable for training on low-resource language applications.
- **Link to Dataset**: You can access the [Flickr30k dataset on Kaggle](https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset) to download images and captions.

> **Note**: You may need to create a Kaggle account to access and download the dataset.

## Requirements

- **Python 3.x**
- **TensorFlow** and **Keras**
- **NumPy**
- **Pandas**
- **NLTK** (for tokenization)
- **OpenCV** (for image processing)

## Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/Ayush-chanchal/Automated-Image-Caption-Generation-System-for-low-resource-Arabic-language.git
   cd arabic-image-captioning
