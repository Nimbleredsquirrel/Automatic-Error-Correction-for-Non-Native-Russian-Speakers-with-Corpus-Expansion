# Automatic Error Correction for Non–Native Russian Speakers with Corpus Expansion

This repository contains examples of models I trained during my experiments, datasets and useful analysis for my diploma. I've uploaded the most representative notebooks from all the work I did.

## What's in here

**simpo-gemma-2.ipynb** - This notebook implements SimPO loss completely from scratch. I also did model quantization here.

**gec_app_file.ipynb** - Shows how to run my GEC application not locally, but through Google Colab using ngrok. Ngrok is a tool that creates a tunnel from your local machine to the internet, so you can share your localhost with others through a public URL.

The repository also includes notebooks for:
- Data preprocessing (the preprocessed RLC-GEC dataset is included too)
- Synthetic data generation (dataset is also attached)
- Data analytics

## Annotation websites

I'm also sharing links to two websites I created for RLC annotation:

**https://squirrelnimble.pythonanywhere.com** - for annotating sentences from the corpus with triple overlap

**https://secondappsite.pythonanywhere.com** - for creating winner-loser pairs to train models using DPO and SimPO
