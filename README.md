# Audio Classification of Accelerating Vehicles

In this project, we used different machine learning techniques to classify vehicles accelerating from a stop. In classification, specific focus was on applying support vector machines, recurrent neural networks, and vanilla fully-connected neural networks to distinguish between anywhere from two to seven different types of vehicles.

[Find the paper here](http://cs229.stanford.edu/proj2019aut/data/assignment_308832_raw/26646848.pdf)

### Documentation
The following are generally helpful scripts for working with electron micrographs:
* Extract_Features_Augmentation.py - Extract a total 678 features from .wav files. Also includes data augmentation strategies
* Data_Statistics.py - Output class statistics for all examples
* FCNN.py - 7-vehicle classification using fully-connected deep learning models
* RNN.py - 7-vehicle classification using sequence models
