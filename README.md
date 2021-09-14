# Cataract_Seg
This repository contains the code of our model in [CATARACTS Semantic Segmentation 2020](https://cataracts-semantic-segmentation2020.grand-challenge.org/). 
<br>
Team Perception are Mobarakol Islam, Bharat Giddwani and Ren Hongliang. They used an encoder-decoder archi- tecture for segmentation. 
The model was adopted from their previous works [5], [6] which contains a residual encoder and a Skip-competitive Spatial and Channel 
Squeeze & Excitation (SC-scSE) decoder as shown in Figure 5-a. The encoder is formed by 5 residual layers as ResNet18 and the corresponding 
decoding block contains convolution, adaptive batch normal- ization [7], SC-scSE, and deconvolution sequentially. The SC- scSE decoder retains 
weak features, excites strong features and performs dynamic spatial and channel-wise feature recal- ibration which makes the network capable of 
better feature learning. They used batch size of 10 for training the proposed model. The model was trained with a learning rate of 0.0001, using 
the Adam optimizer and the momentum and weight decay set as constant to 0.99 and 10−4, respectively. The input images were flipped randomly as a 
part of augmentation. They followed the same data split as [4] for train and validation.<br>

The performance results of our model per task: <br>
Task 1 : 0.8482 <br>
Task 2: 0.7796 <br>
Task 3: 0.7122 <br>


 

