# VarTriage

VarTriage Deep-learning based document classification method

The source of data (The shared datasets are only for testing the classifiers only.)
- UniProtKB/SwissProt: http://www.uniprot.org/downloads

- The GWAS Catalog: https://www.ebi.ac.uk/gwas/docs/file-downloads

- mycoSet: https://github.com/TsangLab/mycoSORT

- PubMed: https://www.ncbi.nlm.nih.gov/pubmed/


Required environment: 
Python 2.7, 
h5py==2.7.1, 
Keras==2.1.5, 
numpy==1.14.1, 
scikit-learn==0.19.1, 
tensorflow-gpu==1.6.0, 
gensim==3.4.0

We built our code based on following sources: 

Convolutional Neural Networks for Sentence Classification (Yoon Kim):
https://github.com/yoonkim/CNN_sentence

Convolutional Neural Networks for Sentence Classification (Alexander Rakhlin): 
https://github.com/alexander-rakhlin/CNN-for-Sentence-Classification-in-Keras

Keras CNN Example - mnist dataset : 
https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py

To run this code, you need to download pre-trained word2vec files.

 - PubMed W2V: http://bio.nlplab.org/ 
 
 - Google News: https://github.com/mmihaltz/word2vec-GoogleNews-vectors


Please contact Kyubum Lee (Kyubum[dot]Lee[at]nih[dot]gov) for questions or comments.
