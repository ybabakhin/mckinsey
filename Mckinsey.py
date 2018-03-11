# Load Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Dense,Embedding,LSTM,Dropout,Bidirectional,Input,BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping,ModelCheckpoint

def main():
    # Load data
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")
    
    # Create labels
    label = train[train.challenge_sequence > 10][['user_id','challenge']]
    label.rename(columns={'challenge':'label'},inplace=True)
    
    # Treat the sequence of challenges as text
    df = train[train.challenge_sequence <= 10].groupby('user_id').challenge.aggregate(lambda x: ' '.join(x)).reset_index()
    
    # Merge Labels
    df = df.merge(label)
    
    # Validation split for early stopping
    df_train, df_validation = train_test_split(df.sample(frac=1,random_state=123), test_size=0.05, random_state=123)
    
    # Load all the challenges
    challenges = pd.read_csv('challenge_data.csv')
    
    # Encode challenges
    encoder = LabelEncoder()
    encoder.fit(challenges['challenge_ID'])
    df_train['brand_id_encoded'] = encoder.transform(df_train.label)
    df_validation['brand_id_encoded'] = encoder.transform(df_validation.label)
    
    # Tokenize text
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(df_train['challenge'])
    
    # Constants
    NB_WORDS = len(tokenizer.word_index)
    MAX_SEQUENCE_LENGTH = 10
    N_CATEGORIES = challenges.shape[0]
    
    # Create sequences
    sequences_train = tokenizer.texts_to_sequences(df_train['challenge'])
    sequences_validation = tokenizer.texts_to_sequences(df_validation['challenge'])
    
    # Pad sequences
    x_train = pad_sequences(sequences_train, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
    x_validation = pad_sequences(sequences_validation, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
    
    # Set Labels
    y_train = df_train['brand_id_encoded'].values
    y_validation= df_validation['brand_id_encoded'].values
    
    # NN architecture
    def get_model(path='',lr=0.001):
        adam = Adam(lr=lr)
        inp = Input(shape=(MAX_SEQUENCE_LENGTH, ))
        x = Embedding(NB_WORDS,256)(inp)
        x = BatchNormalization()(x)
        x = Bidirectional(LSTM(128, dropout=0.1, recurrent_dropout=0.1))(x)
        x = Dropout(0.4)(x)
        x = Dense(N_CATEGORIES, activation="softmax")(x)
        model = Model(inputs=inp, outputs=x)
        if path != '':
            model.load_weights(path)
        model.compile(loss='sparse_categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
        return model
    
    # Initialize the model
    model = get_model()
    
    # Model callbacks
    path = 'best_model_weights'
    es_callback = EarlyStopping(monitor="val_loss", patience=5)
    mc_callback = ModelCheckpoint('{}.hdf5'.format(path), monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=True, mode='auto', period=1)
    callbacks = [es_callback,mc_callback]
    
    # Fit the model
    model.fit(x_train,
              y_train,
              epochs=100,
              batch_size=1024,
              validation_data=(x_validation, y_validation),
              callbacks = callbacks
             )
    
    # Load best weights
    model = get_model('{}.hdf5'.format(path))
    
    # Test preprocessing
    def padding(text):
        return pad_sequences(tokenizer.texts_to_sequences(text), maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
    test_text = test[test.challenge_sequence <= 10].groupby('user_id').challenge.aggregate(lambda x: ' '.join(x)).reset_index()
    x_test = padding(test_text.challenge)
    
    # Get top 3 predictions for each user
    pred = model.predict(x_test,batch_size=2048)
    pred = pred.argsort(axis=1)[:,-3:][:,::-1]
    
    # Write Predictions
    df_list = []
    for i in range(3):
        test_11 = test_text[['user_id']]
        test_11['user_sequence'] = test_11.user_id.astype(str) + '_'+str(i+11)
        test_11['challenge'] = encoder.inverse_transform(pred[:,i])
        df_list.append(test_11[['user_sequence','challenge']])
    pd.concat(df_list).to_csv('bes_submission.csv',index=False)

if __name__ == '__main__':
    main()