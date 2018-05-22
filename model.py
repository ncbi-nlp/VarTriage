import os, pickle, time
import sys
reload(sys)
sys.setdefaultencoding('utf8')
import numpy as np
from keras import metrics, regularizers
from keras.layers import Input, Dense, Embedding, Conv2D, MaxPool2D
from keras.layers import Reshape, Flatten, Dropout, Concatenate
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.models import Model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from datetime import datetime
from sklearn.metrics import f1_score, precision_score, recall_score
from gensim.models import KeyedVectors, Word2Vec
import keras.backend as K
from data_helpers import load_data_and_labels, load_word2vec

###############
# Optional for GPU Selection Instead, you can also do "CUDA_VISIBLE_DEVICES=0"
# import GPUtil
# GPUtil.showUtilization()
# DEVICE_ID_LIST = GPUtil.getFirstAvailable() # Get the first available GPU
# DEVICE_ID = DEVICE_ID_LIST[0] # grab first element from list
# os.environ["CUDA_VISIBLE_DEVICES"] = str(DEVICE_ID) # Set CUDA_VISIBLE_DEVICES to mask out all other GPUs than the first available device id
# print('Device ID (unmasked): ' + str(DEVICE_ID)) # Since all other GPUs are masked out, the first available GPU will now be identified as GPU:0

###########################################################################
# Inputs

#GWAS
file_tag = "GWAS"
pos_file = "./data/GWAS/gwas_before2017_train_pos.txt"
neg_file = "./data/GWAS/gwas_before2017_train_neg.txt"
BASE_DIR = ''
keras_checkpoint_dir = os.path.join(BASE_DIR, 'checkpoints')
W2V_file_addr = os.path.join(BASE_DIR, 'W2V', 'PubMed-and-PMC-w2v.bin')  # Pre-trained Word2vec file
# W2V_file_addr = os.path.join(BASE_DIR, 'W2V', 'wikipedia-pubmed-and-PMC-w2v.bin') # Pre-trained Word2vec file

# #UNIPROTKB
# file_tag = "UNIP"
# pos_file = "./data/UNIP/unip_before2017_train_pos.txt"
# neg_file = "./data/UNIP/unip_before2017_train_neg.txt"
# BASE_DIR = ''
# keras_checkpoint_dir = os.path.join(BASE_DIR, 'checkpoints')
# W2V_file_addr = os.path.join(BASE_DIR, 'W2V', 'PubMed-and-PMC-w2v.bin') # Pre-trained Word2vec file

# #MYCOSET
# file_tag = "MYCO"
# pos_file = "./data/MYCO/mycoSet_PMIDs_POS_toText_train.txt"
# neg_file = "./data/MYCO/mycoSet_PMIDs_NEG_toText_train.txt"
# BASE_DIR = ''
# keras_checkpoint_dir = os.path.join(BASE_DIR, 'checkpoints')
# #W2V_file_addr = os.path.join(BASE_DIR, 'W2V', 'PubMed-and-PMC-w2v.bin') # Pre-trained Word2vec file
# W2V_file_addr = os.path.join(BASE_DIR, 'W2V', 'wikipedia-pubmed-and-PMC-w2v.bin') # Pre-trained Word2vec file

###########################################################################
#Parameters
max_sequence_length = 1000
embedding_dim = 200 # however, if there is a pre-trained word2vec file, follow the dimensionality of the file.
filter_sizes = [3,4,5]
num_filters = 2048
hidden_dims = 100
drop = 0.5
validation_split = 0.1  # Use this much for validation, others for training
learning_rate = 1e-5
epochs = 50
batch_size = 50
max_num_words = 6000000


###########################################################################
def main():
    current_time = str(datetime.now().strftime('%Y%m%d_%H%M%S'))
    print(current_time)

    print('Loading data')
    texts, labels = load_data_and_labels(pos_file, neg_file)

    if W2V_file_addr is not None:
        print('Loading Word2Vec')
        # embeddings_index, embedding_dim = load_word2vec_nonbinary(W2V_file_addr)
        embeddings_index, embedding_dim = load_word2vec(W2V_file_addr)
        print('Found %s word vectors.' % len(embeddings_index))
    else :
        embeddings_index = None
        embedding_dim = 300

    checkpoint_dir = os.path.join(keras_checkpoint_dir, file_tag, current_time)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # vectorize the text samples into a 2D integer tensor
    print('Tokenizing the texts')
    tokenizer = Tokenizer(num_words=max_num_words, lower=False)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)

    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))
    tokenizer_addr = os.path.join(checkpoint_dir, 'tokenizer.pickle')
    with open(tokenizer_addr, 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print('Tokenizer is saved as %s' % tokenizer_addr)
    print('Padding Sequences')
    data = pad_sequences(sequences, maxlen=max_sequence_length)

    print('Shape of data tensor:', data.shape)
    print('Shape of label tensor:', labels.shape)

    # split the data into a training set and a validation set
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]
    num_validation_samples = int(validation_split * data.shape[0])

    x_train = data[:-num_validation_samples]
    y_train = labels[:-num_validation_samples]
    x_val = data[-num_validation_samples:]
    y_val = labels[-num_validation_samples:]

    # prepare embedding matrix
    num_words = min(max_num_words, len(word_index) + 1)
    if W2V_file_addr is not None:
        embedding_matrix = np.zeros((num_words, embedding_dim))
        for word, i in word_index.items():
            if i >= max_num_words:
                continue
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
    else :
        embedding_matrix = np.zeros((num_words, embedding_dim))

    # Build model
    model_input = Input(shape=(max_sequence_length,), dtype='int32')

    z = Embedding(num_words,
                  embedding_dim,
                  weights=[embedding_matrix],
                  input_length=max_sequence_length,
                  trainable=False)(model_input)
    z = Reshape((max_sequence_length, embedding_dim, 1))(z)

    # Convolutional block
    conv_blocks = []
    for sz in filter_sizes:
        conv = Conv2D(num_filters, kernel_size=(sz, embedding_dim), padding='valid', activation='relu')(z)
        conv = MaxPool2D(pool_size=(max_sequence_length - sz + 1, 1), strides=(1, 1), padding='valid')(conv)
        conv = Flatten()(conv)
        conv_blocks.append(conv)
    z = Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]
    #z = Dense(hidden_dims, kernel_regularizer=regularizers.l2(0.0001),activity_regularizer=regularizers.l1(0.0001), activation="relu")(z)
    z = Dense(hidden_dims, activation="relu")(z)
    z = Dropout(drop)(z)
    model_output = Dense(units=2, activation='softmax')(z)

    adam = Adam(lr=learning_rate, decay=1e-6)
    # adam = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.00001)
    # adam = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0001)

    model = Model(model_input, model_output)
    model.summary()

    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['mse', 'acc'])  # <==<==biogpu12-1
    #model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
    #model.compile(optimizer=adam, loss='mean_squared_error', metrics=['mse', 'acc']) # <== best <==biogpu12-2


    # Log Dir
    log_dir = os.path.join(checkpoint_dir, '../')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    f_log = open(os.path.join(log_dir, 'result_logs_' + file_tag + '.txt'), 'a')
    print('Log File: %s' % os.path.join(os.getcwd(), log_dir, 'result_logs_' + file_tag + '.txt'))
    f_log.write("\n")
    f_log.write(current_time)
    f_log.write("\n")
    f_log.write('pos_file: %s\n' % pos_file)
    f_log.write('neg_file: %s\n' % neg_file)
    f_log.write('checkpoint_folder: %s\n' % checkpoint_dir)
    f_log.write('embedding_dim = %s\n' % embedding_dim)
    f_log.write('W2V_file_addr = %s\n' % W2V_file_addr)
    f_log.write('filter_sizes = %s\n' % filter_sizes)
    f_log.write('num_filters = %s\n' % num_filters)
    f_log.write('hidden_dims = %s\n' % hidden_dims)
    f_log.write('drop = %s\n' % drop)
    f_log.write('validation_split = %s\n' % validation_split)
    f_log.write('learning_rate = %s\n' % learning_rate)
    f_log.write('epochs = %s\n' % epochs)
    f_log.write('batch_size = %s\n' % batch_size)
    f_log.write('max_num_words = %s\n' % max_num_words)
    f_log.write('max_sequence_length = %s\n' % max_sequence_length)
    f_log.flush()

    checkpoint = ModelCheckpoint(os.path.join(checkpoint_dir, file_tag + "_" + current_time + '_weights.{epoch:03d}-{val_acc:.4f}.hdf5'), monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

    print("Traning Model...")
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=2, callbacks=[checkpoint],
              validation_data=(x_val, y_val))  # starts training
    pred_results = model.predict(x_val)
    f1_res = f1_score(y_val.argmax(axis=1), pred_results.argmax(axis=1))
    precision_res = precision_score(y_val.argmax(axis=1), pred_results.argmax(axis=1))
    recall_res = recall_score(y_val.argmax(axis=1), pred_results.argmax(axis=1))
    
    print("\n")
    print("F1:\t%s" % f1_res)
    print("Precision:\t%s" % precision_res)
    print("recall:\t%s" % recall_res)

    f_log.write("F1:\t%s\n" % f1_res)
    f_log.write("Precision:\t%s\n" % precision_res)
    f_log.write("recall:\t%s\n" % recall_res)
    f_log.write("\n\n")
    f_log.flush()
    f_log.close()

    model.save(os.path.join(checkpoint_dir, 'final_model.h5') )  # creates a HDF5 file 'my_model.h5'
    del model  # deletes the existing model

    print("For evaluation, please use the following checkpoint : %s " % checkpoint_dir)

if __name__ == '__main__':
    main()
