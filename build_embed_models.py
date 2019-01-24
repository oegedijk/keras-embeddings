"""

Build Deep learning models using entity embeddings loaded from disk.


"""

import numpy as np
import pandas as pd

from keras.models import Model, Sequential
from keras.layers import Input, Dense, Dropout, Activation
from keras.layers import Concatenate, Reshape, SpatialDropout1D
from keras.layers.embeddings import Embedding
from keras import regularizers

print("Loading processed data...")
d  = load_training_data()

print("Loading validation data...")
dval = load_validation_data()

target = 'profit'

y = np.log(1.0+d[profit].astype(float))
yval = np.log(1.0+dval[profit].astype(float))

print("loading embeddings")
embedding_dict = pickle.load(open(str('embedding.dict'), "rb"))

print("Loading feature lists...")
cat_features = saml_load_feature_list('categorical_columns')
num_features = saml_load_feature_list('numeric_columns')


def embed_data_preprocess(X_train, X_val, cat_features, num_features):
    """
    return input data as a list with the categorical features and the
    numeric features in the right order.
    """
    input_list_train = []
    input_list_val = []

    # First add all the categorical features
    for col in cat_features:
        input_list_train.append(X_train[col].values)
        input_list_val.append(X_val[col].values)

    # then add the numerical columns in at the end:
    input_list_train.append(X_train[num_features].values)
    input_list_val.append(X_val[num_features].values)

    return input_list_train, input_list_val


def build_embedding_model(d, y, dval, yval,
                          cat_features, num_features, embedding_dict):
    """
    Return trained Keras model with entity embeddings and training history.
    """
    inputs = []
    embeddings = []

    # load the entity embeddings from embedding_dict
    for col in cat_features:
        # Use the same cardinality and embedding_dim as before
        # Could also read these from the embedding_dict directly
        cardinality = int(np.ceil(d[col].nunique() + 2))
        embedding_dim = max(min((cardinality)//2, 50),2)
        print(f'{col}: cardinality : {cardinality} and embedding dim: {embedding_dim}')

        col_inputs = Input(shape=(1,))
        # Now we set the weights to the pre-trained embeddings in the
        # embedding_dict.
        # If you set trainable=False, then these embeddings will not be
        # updated during training.
        embedding = Embedding(cardinality, embedding_dim, input_length=1,
                              weights = [embedding_dict[col]], name=col+"_embed",
                              trainable=False)(col_inputs)
        embedding = Reshape(target_shape=(embedding_dim,))(embedding)

        inputs.append(col_inputs)
        embeddings.append(embedding)

    # load the numeric features
    input_numeric = Input(shape=(len(num_features),))
    inputs.append(input_numeric)

    # Put a layer in front of the numeric features with 5 neurons less then
    # the number of inputs.
    numeric_first_layer = Dense(len(num_features)-5)(input_numeric)

    # append the embeddings and the numeric features together.
    embeddings.append(numeric_first_layer)

    # Now add a neural network on top:
    x = Concatenate()(embeddings)
    x = Dense(1024, activation='relu',
              kernel_regularizer=regularizers.l2(0.02))(x)
    x = Dropout(.5)(x)
    x = Dense(512, activation='relu',
              kernel_regularizer=regularizers.l2(0.02))(x)
    x = Dropout(.5)(x)
    x = Dense(512, activation='relu',
              kernel_regularizer=regularizers.l2(0.02))(x)
    outputs = Dense(1)(x)

    # define and compile the model:
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss= "mean_squared_logarithmic_error" ,
                  optimizer="adam",
                  metrics=["mse", "mae"])

    # prepare the data:
    X_train, X_val = embed_data_preprocess(d, dval, cat_features, num_features

    history = model.fit(X_train, y.values,
                        validation_data = (X_val, yval.values),
                        batch_size=1024, epochs=15, callbacks=[clr])

    return model, history


embedding_model, history = build_embedding_model(d, y, dval, yval,
                                cat_features, num_features, embedding_dict)
