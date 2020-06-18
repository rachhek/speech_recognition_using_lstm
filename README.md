# speech_recognition_using_lstm
This project trained a neural network model using LSTM RNN with 54 hours of speech from 6 different languages to classify speech samples.
LSTM RNN = Long Short Term Memory Recurrent Neural Networks

## Introduction
- To determine which language is being spoken in a speech sample
- Humans recognize it through perceptual process inherent in auditory system
- The aim is to replicate human ability through computational means
- How to scientifically distinguish diverse spoken lanugages in the world to correctly classify speech samples?
- LSTM RNNS is an excellent choice for classifying a speech sample because they can effectively exploit temporal dependencies in acoustic data and they reportedly perfom better than DNN

## Dataset
- Subset of Mozilla Common Voice speech dataset (voice.mozilla.org)
- mp3 format converted to 16kHz waveform
- Volume normalized to -3dbfs
- 54 hours of speech
- 6 different languages

## Methodology
- Step 1: Data preprocessing and feature extraction using MFCC
- Step 2: Classifier training using CNN and LSTM
- Step 3: Model Evaluation

## Technologies used
- Python
- Keras Tensorflow

## Step 1: Data preprocessing
- Split .wav files to equal length audio of 3secs
- Generate MFCC features with 1 sec sliding window
- Normalize the data using MinMax scaler
- Label the audios

```
(rate,sig) = wav.read(wav_file)
mfcc_feat = mfcc(sig,rate)
scaler = scaler.fit(mfcc_feat)
normalized = scaler.transform(mfcc_feat)
```

## Step 2: Classifier training using CNN and LSTM
- Built using Keras and Tensorflow
- One LSTM layer with 200 units
- One dense layer with 6 units
- Softmax activation
- Learning rate: 0.001
- Early stopping using val_loss

```
#Setup constants
EPOCHS = 100
BATCH_SIZE = 40
SAMPLES_PER_EPOCH = len(range(0, len(X_train_array), BATCH_SIZE))

#Declaring callbacks for early stopping
callbacks = [
             EarlyStopping(monitor='val_loss', min_delta=0.01, patience=5,mode = 'min')
            ]


#Creating a LSTM with 200 units
model= Sequential()
model.add(LSTM(200,input_shape=(299,13),return_sequences=False))
model.add(Dense(6,activation='softmax'))
model.compile(optimizer=Adam(amsgrad=True, lr=0.001),loss='categorical_crossentropy',metrics=['accuracy'])

#Print the model summary
model.summary()

#Fit the model
history = model.fit(X_train_array, 
                    Y_train, 
                    epochs=EPOCHS,
                    callbacks = callbacks,
                    batch_size=BATCH_SIZE,
                    validation_data=(X_test, Y_test),
                    verbose=1,
                    shuffle=True
                   )

#Model Summary
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
lstm_12 (LSTM)               (None, 200)               171200    
_________________________________________________________________
dense_11 (Dense)             (None, 6)                 1206      
=================================================================
Total params: 172,406
Trainable params: 172,406
Non-trainable params: 0
```

## Step 3: Model evaluation
Accuracy of upto 83% was achieved. 

### Project collaborators
This project was done together with @alvarorgaz for a course called Automatic Speech Recognition at Aalto University.
