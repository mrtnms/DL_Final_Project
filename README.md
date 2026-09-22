# Course Deep Learning: final project

Developed a meme generator that takes any image as input and generates a "funny" caption for that image. 
Training data consists out of 900K memes that cover 300 popular templates, each with thousands of captions. 

## Architecture

### Stage 1: ResNet encoder
- Input: image
- Output: feature vector

### Stage 2: LSTM
- Input: feature vector
- Output: caption
- Loss function: cross-entropy
