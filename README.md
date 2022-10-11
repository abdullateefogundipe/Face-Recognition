# Face-Recognition
We create a face recognition/verification model that can be trained on the faces of multiple people. This model can then be used to predict which of the saved identities a new image belongs to.

Dlib and OpenCV are used to extract and align the face from images.

A CNN model is created which can extract special embedding features from the face

Images of N number of people are used to train the CNN model. For each person, multiple training images are used.

A second machine learning model is trained with the embedding obtained from the CNN model for each image. The features of the data are the extracted embeddings while the label is the person's name or index.

With these, the identity of the face in new images can be determined by extracting the embedding from each image, and predicting the label using the second machine learning model.

SideNote: You should also know that faces having embeddings with small euclidean distance from eachother are more likely to be the same person than those having larger euclidean distance

Face Recognition training & Predict jupyter notebook extracts relevant function from Model.py and Image_preprocessing.py
Model.py contains the CNN model used to extract embeddings from Faces while the Image_preprocessing.py contains functions for cropping out and straightening the face from any image

The Predict_model.py can be modified to train new data containing new identities. 
