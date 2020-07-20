## Scientific Paper Classifier

### Purpose

To assign a scientific paper based on its abstract to one of two similar arXiv categories: `hep-ph` or `hep-th`. 
A classification accuracy of 86% is reached, without extensive hyperparameter tuning.


### Description of files

__[import-metadata.py](import-metadata.py)__ -
A python script for harvesting meta-data of 100k+ papers from the two arXiv categories. The data in XML format, returned by the OAI-PMH interface, is stored in text files.

__[paper-classifier.py](paper-classifier.py)__ -
The main python script for preprocessing the meta-data in XML format into features and labels, involving stemming, vectorisation and dimensionality reduction. The machine learning classifier algorithm Support Vector Machine is trained and its performance on the test set is evaluated.

__[utilities.py](utilities.py)__ -
Python script with custom helper functions for paper-classifier.py
