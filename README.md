# NLP2-TweetGenerator

As part of our NLP (Natural Language Processing) course, we carried out a project with an open theme.
We chose to focus on generating tweet comments that can manipulate public opinion by introducing topics and ideologies into trends.

This project was carried out in groups of 3 and allowed me to obtain an average grade of 15 in this subject (NLP2).



## 1. The project instructions:

Goals
 - Find an objective/a problem - has to be related to NLP (see project ideas)
 - Explain the motivation
 - Find the right NLP models/tools/datasets
 - Explain how you selected them
 - Build a global architecture to solve the problem
 - Explain the global architecture of your proposal
 - Either train or use a trained model + find a quantitative way to evaluate performance.
 - Explain how the models work.
 - Build a demonstrator (streamlit, gradio, other?).


## 2. Project Structure
```
.
├── doc
│   └── presentation_diapo.pdf
├── Makefile
├── README.md
└── src
    ├── notebook
    │   └── NLP.ipynb
    └── streamlit
        └── app.py
``

 - /doc/presentation_diapo.pdf  -->  slideshow presentation of our project.
 - /src/notebook/NLP.ipynb      -->  contains the core implementation of the project, focusing on data preprocessing, model training, and evaluation using NLP techniques. It provides a detailed step-by-step workflow, from dataset preparation to final results.
 - /src/streamit/app.py         -->  a Streamlit application that allows you to interact with the project by selecting tweets and generating realistic Twitter-style comments.
 - Makefile                     -->  to automate the installation of dependencies and download the required file.


## 3. Usage Instructions

First, launch the Makefile to install the elements necessary to test this code:
``` make ```

### Notebook
Just run the entire notebook cells. And follow the detailed explanation of it. [NLP.ipynb](./src/notebook/NLP.ipynb)

### Streamlit
Pour lancer le streamlit :
``` streamlit run src/streamlit/app.py ```
Select the tweet of your choice or create the one you want.
Select the number of comments to generate.


## 4. Model Summary

The Phi-3-Mini-4K-Instruct is a 3.8B parameters, lightweight, state-of-the-art open model trained with the Phi-3 datasets that includes both synthetic data and the filtered publicly available websites data with a focus on high-quality and reasoning dense properties. The model belongs to the Phi-3 family with the Mini version in two variants 4K and 128K which is the context length (in tokens) that it can support. [Here](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct)
