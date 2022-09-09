# import the necessary packages
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pickle

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.optimizers import Adam
from xgboost import XGBClassifier
from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB
import os

import types
import tempfile
import keras.models
from flask import flash

def make_keras_picklable():
    def __getstate__(self):
        model_str = ""
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
            keras.models.save_model(self, fd.name, overwrite=True)
            model_str = fd.read()
        d = { 'model_str': model_str }
        return d

    def __setstate__(self, state):
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
            fd.write(state['model_str'])
            fd.flush()
            model = keras.models.load_model(fd.name)
        self.__dict__ = model.__dict__


    cls = keras.models.Model
    cls.__getstate__ = __getstate__
    cls.__setstate__ = __setstate__

def create_mlp_model(optimizer='adam', neuron_number=50, lr=0.001, class_number=10):
    # Build function for keras/tensorflow based multi layer perceptron implementation
    model = Sequential()
    model.add(Dense(neuron_number, input_dim=128, activation='relu'))
    model.add(Dense(neuron_number, activation='relu'))
    model.add(Dense(class_number, activation='softmax'))
    optimizer = Adam(lr=lr)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model



def train_mlp_model(embeddings_path = "", classifier_model_path = "", label_encoder_path = ""):
    # Trains a MLP classifier using embedding file from "embeddings_path", 
    # then saves the trained model as "classifier_model_path" and 
    # label encoding as "label_encoder_path".

    #make_keras_picklable()
    # Load the face embeddings
    print("[INFO] loading face embeddings...")
    data = pickle.loads(open(embeddings_path, "rb").read())
    
    # Encode the labels
    print("[INFO] encoding labels...")
    le = LabelEncoder()
    labels = le.fit_transform(data["names"])
    class_number = len(set(labels))
    
    # Reshape the data
    embedding_mtx = np.zeros([len(data["embeddings"]),len(data["embeddings"][0])])
    for ind in range(1,len(data["embeddings"])):
        embedding_mtx[ind,:] = data["embeddings"][ind]
        
    # Train the model used to accept the 128-d embeddings of the face and
    # then produce the actual face recognition
    print("[INFO] training model...")

    recognizer = KerasClassifier(build_fn=create_mlp_model, epochs=450, batch_size=64, verbose=1, neuron_number = 32, lr = 1e-3, class_number = class_number)

    recognizer.fit(embedding_mtx, labels)
    
    print("[INFO] saving model...")
    #Write the actual face recognition model to disk as pickle
    with open(classifier_model_path, "wb") as write_file:
        pickle.dump(recognizer, write_file)

    #recognizer.save(classifier_model_path)

    # Write the label encoder to disk as pickle
    with open(label_encoder_path, "wb") as write_file:
        pickle.dump(le, write_file)
        
def train_svm_model(embeddings_path, classifier_model_path, label_encoder_path):
    # Trains a SVM classifier using embedding file from "embeddings_path", 
    # then saves the trained model as "classifier_model_path" and 
    # label encoding as "label_encoder_path".
    
    # Load the face embeddings
    flash("[INFO] loading face embeddings...")
    data = pickle.loads(open(embeddings_path, "rb").read())
    
    # Encode the labels
    flash("[INFO] encoding labels...")
    le = LabelEncoder()
    #print(data["names"])
    labels = le.fit_transform(data["names"])
    
    # Train the model used to accept the 128-d embeddings of the face and
    # then produce the actual face recognition
    flash("[INFO] training model...")
    recognizer = SVC(C=100, kernel="poly", probability=True)
    recognizer.fit(data["embeddings"], labels)
    
    flash("[INFO] saving model...")
    # Write the actual face recognition model to disk as pickle
    with open(classifier_model_path, "wb") as write_file:
        pickle.dump(recognizer, write_file)
    
    # Write the label encoder to disk as pickle
    with open(label_encoder_path, "wb") as write_file:
        pickle.dump(le, write_file)

        
def train_nb_model(embeddings_path = "", classifier_model_path = "", label_encoder_path = ""):
    # Trains a NB classifier using embedding file from "embeddings_path", 
    # then saves the trained model as "classifier_model_path" and 
    # label encoding as "label_encoder_path".
    
    # Load the face embeddings
    flash("[INFO] loading face embeddings...")
    data = pickle.loads(open(embeddings_path, "rb").read())
    
    # Encode the labels
    flash("[INFO] encoding labels...")
    le = LabelEncoder()
    labels = le.fit_transform(data["names"])
    
    # Train the model used to accept the 128-d embeddings of the face and
    # then produce the actual face recognition
    flash("[INFO] training model...")
    recognizer = GaussianNB()
    recognizer.fit(data["embeddings"], labels)
    
    flash("[INFO] saving model...")
    # Write the actual face recognition model to disk as pickle
    with open(classifier_model_path, "wb") as write_file:
        pickle.dump(recognizer, write_file)
    
    # Write the label encoder to disk as pickle
    with open(label_encoder_path, "wb") as write_file:
        pickle.dump(le, write_file)


def train_xgboost_model(embeddings_path="", classifier_model_path="", label_encoder_path=""):
    # Trains a SVM classifier using embedding file from "embeddings_path",
    # then saves the trained model as "classifier_model_path" and
    # label encoding as "label_encoder_path".

    # Load the face embeddings
    flash("[INFO] loading face embeddings...")
    data = pickle.loads(open(embeddings_path, "rb").read())

    # Encode the labels
    flash("[INFO] encoding labels...")
    le = LabelEncoder()
    labels = le.fit_transform(data["names"])

    embedding_mtx = np.zeros([len(data["embeddings"]), len(data["embeddings"][0])])
    for ind in range(1, len(data["embeddings"])):
        embedding_mtx[ind, :] = data["embeddings"][ind]

    # Train the model used to accept the 128-d embeddings of the face and
    # then produce the actual face recognition
    flash("[INFO] training model...")
    recognizer = XGBClassifier()
    #recognizer = SVC(C=1, kernel="linear", probability=True)
    recognizer.fit(embedding_mtx, labels)

    flash("[INFO] saving model...")
    # Write the actual face recognition model to disk as pickle
    with open(classifier_model_path, "wb") as write_file:
        pickle.dump(recognizer, write_file)

    # Write the label encoder to disk as pickle
    with open(label_encoder_path, "wb") as write_file:
        pickle.dump(le, write_file)

# embeddings_path = "/home" + os.sep + "andrew" + os.sep + "diploma" + os.sep + "flask" + os.sep + "my_models" + os.sep + "20_30.pickle"
# classifier_model_path = "/home" + os.sep + "andrew" + os.sep + "diploma" + os.sep + "flask" + os.sep + "my_models" + os.sep + "20_30REC-100p1.pickle"
# label_encoder_path = "/home" + os.sep + "andrew" + os.sep + "diploma" + os.sep + "flask" + os.sep + "my_models" + os.sep + "20_30LAB-100p1.pickle"
# train_svm_model(embeddings_path, classifier_model_path, label_encoder_path)
