



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
<!-- [![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url] -->



<!-- PROJECT LOGO
<br />
<div align="center">
  <a href="https://github.com/othneildrew/Best-README-Template">
    <img src="images/logo.png" alt="Logo" width="80" height="80">
  </a>

  <h3 align="center">Best-README-Template</h3>

  <p align="center">
    An awesome README template to jumpstart your projects!
    <br />
    <a href="https://github.com/othneildrew/Best-README-Template"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/othneildrew/Best-README-Template">View Demo</a>
    ·
    <a href="https://github.com/othneildrew/Best-README-Template/issues">Report Bug</a>
    ·
    <a href="https://github.com/othneildrew/Best-README-Template/issues">Request Feature</a>
  </p>
</div> -->



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

<!-- [![Product Name Screen Shot][product-screenshot]](https://example.com) -->

Our project compares traditional machine learning techniques to modern approaches for audio genre classification.

We test the following traditional machine learning methods:
FFNN (Feed Forward Neural Network (no attention, convoltion, or anything fancy)) (sanity check) (use pytorch) (Omeed)
SVM (Felipe)
Decision tree (Kunal)
naive bayes 
K-nearest neighbors
some kind of pipeline 
Ensemble classifiers
cross validation
Regression
2. Clustering (DBSCAN, Hierarchical Clustering, K-Means)

We test the following modern machine learning methods:
1. VGG16, Inception, AlexNet, or something more recent

<!-- GETTING STARTED -->
## Getting Started

This is an example of how you may give instructions on setting up your project locally.
To get a local copy up and running follow these simple example steps.

virtualenv music_genre_env
source music_genre_env/bin/activate
pip3 install -r requirements.txt


<!-- # to handle m1 issue: 
conda config --add channels conda-forge
pip install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu


conda create -y --name pygenreaudio python=3.8
conda install --force-reinstall -y -q --name pygenreaudio -c conda-forge --file requirements.txt
conda activate pygenreaudio -->
<!-- 

Create a conda environment: conda create --name pygenre python=3.8
                            conda install  -n pygenre pip
                        <!-- conda install pip -->
<!-- activate environment:   WINDOWS: activate py35
                        LINUX, macOS: conda activate pygenre
run requirements.txt file: pip3 install -r requirements.txt
if on mac m1: pip install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu
else:
  pip install torchaudio --> -->

<!-- ABOUT THE PROJECT -->
## Dataset Used

We use the Free Music Archive (FMA), dataset containing 106,574 tracks with associated MP3 files, artist information, and genre. The dataset is over 100 Gigabytes, and therefore, we use FMA Small, a dataset of 8,000 tracks of 30s, 8 balanced genres (GTZAN-like) (7.2 GiB). The dataset has several issues. I dissagree with several of the genre labels used.

### Prerequisites

For reproducibility, this project relies on the FMA dataset available at https://github.com/mdeff/fma.


## Evaluation
We evaluate the genre classification using both traditional error metrics including recall, precision, accuracy. One challenge with evaluating accuracy is that a song can be multiple genres. A song, for example, can be both acoustic and pop. For this reason,
we also evaluate top-k accuracy, or the chance that the out of the top-k most likely predicted genres, one of them is the labled genre. 

# Info to Group:
Look at this for how to do data augmentation. https://music-classification.github.io/tutorial/part3_supervised/tutorial.html

<!-- CONTACT -->
## Contact

Alex Chandler - alex.chandler@utexas.edu
