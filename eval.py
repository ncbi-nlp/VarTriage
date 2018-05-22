import os, pickle
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from data_helpers import load_data, load_data_eval,load_data_and_labels
from datetime import datetime
from sklearn.metrics import f1_score, precision_score, recall_score

###############
# # Optional for GPU Selection Instead, you can also do "CUDA_VISIBLE_DEVICES=0"
# import GPUtil
# GPUtil.showUtilization()
# DEVICE_ID_LIST = GPUtil.getFirstAvailable() # Get the first available GPU
# DEVICE_ID = DEVICE_ID_LIST[0] # grab first element from list
# os.environ["CUDA_VISIBLE_DEVICES"] = str(DEVICE_ID) # Set CUDA_VISIBLE_DEVICES to mask out all other GPUs than the first available device id
# print('Device ID (unmasked): ' + str(DEVICE_ID)) # Since all other GPUs are masked out, the first available GPU will now be identified as GPU:0

###########################################################################

# #data input GWAS
# file_tag = "GWAS"
# pos_file = "./data/GWAS/gwas_before2017_test_pos.txt"
# neg_file = "./data/GWAS/gwas_before2017_test_neg.txt"
# checkpoint_folder = './checkpoints/GWAS/20180516_162442/' #Use your own checkpoint folder
# checkpoint_file = os.path.join(checkpoint_folder, 'final_model.h5')
# tokenizer_file = os.path.join(checkpoint_folder, 'tokenizer.pickle')

# #data input MYCO
# file_tag = "MYCO"
# pos_file = "./data/MYCO/mycoSet_PMIDs_POS_toText_test.txt"
# neg_file = "./data/MYCO/mycoSet_PMIDs_NEG_toText_test.txt"
# checkpoint_folder = './checkpoints/MYCO/20180521_105845/'
# checkpoint_file = os.path.join(checkpoint_folder, 'final_model.h5')
# tokenizer_file = os.path.join(checkpoint_folder, 'tokenizer.pickle')


batch_size = 50
print('Loading data')
current_time = str(datetime.now().strftime('%Y%m%d_%H%M%S'))
print(current_time)
log_dir = os.path.join('./logs')
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
f_log = open(os.path.join(log_dir, 'result_logs_'+file_tag+'.txt'), 'a')
print('Log File: %s' % os.path.join(os.getcwd(), log_dir, 'result_logs_'+file_tag+'.txt'))
f_log.write("\n")
f_log.write(current_time)
f_log.write("\n")
f_log.write('pos_file: %s\n' % pos_file)
f_log.write('neg_file: %s\n' % neg_file)
f_log.write('checkpoint_folder: %s\n' % checkpoint_folder)

texts, labels = load_data_and_labels(pos_file, neg_file)

pmids = []
for i in range(0, len(texts)):
    #print(texts[i])
    pmidtemp = texts[i].split()[0]
    pmids.append(pmidtemp)

model = load_model(checkpoint_file)
input_seq_length = model.input_shape[1]
EMBEDDING_DIM = model.input_shape[0]
MAX_SEQUENCE_LENGTH = model.input_shape[1]

# loading tokenizer
with open(tokenizer_file, 'rb') as handle:
    tokenizer = pickle.load(handle)
sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

x_test = data
y_test = labels

score, mse, acc = model.evaluate(data, y_test, batch_size, verbose=2)
pred_results = model.predict(data)
print("===============")
print("loss:\t%s" % score)
print("mse:\t%s" % mse)
print("accuracy:\t%s" % acc)
f1_res = f1_score(y_test.argmax(axis=1), pred_results.argmax(axis=1))
precision_res = precision_score(y_test.argmax(axis=1), pred_results.argmax(axis=1))
recall_res = recall_score(y_test.argmax(axis=1), pred_results.argmax(axis=1))
print("F1:\t%s" % f1_res)
print("Precision:\t%s" % precision_res)
print("recall:\t%s" % recall_res)
print("===============")

print("Writing on files...")
f_listoutput_file = pos_file+'_results.txt'
f_listoutput = open(f_listoutput_file, 'w')
f_listoutput.writelines("PMID\tAnswer\tPred\tNegScore\tPosScore\n")
for x in range(0, len(pred_results)):
        f_listoutput.writelines("%s\t%d\t%d\t%.5f\t%.5f\n" % (pmids[x], y_test.argmax(axis=1)[x], pred_results.argmax(axis=1)[x], pred_results[x][0],pred_results[x][1]))

f_listoutput.write("\n")
f_listoutput.close()
print("Evaluation results are saved at %s" % f_listoutput_file)

f_log.write("Loss:     \t%s\n" % score)
f_log.write("MSE:      \t%s\n" % mse)
f_log.write("Accuracy:      \t%s\n" % acc)
f_log.write("F1:       \t%s\n" % f1_res)
f_log.write("Precision:\t%s\n" % precision_res)
f_log.write("recall:   \t%s\n" % recall_res)
f_log.write("\n")
f_log.close()
print("Logs are saved at %s" % log_dir)
print("\n")

