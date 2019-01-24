"""
Train the entity embeddings for every categorical variable and save them to disk.

We train a model with only entity embeddings (so no accompanying numeric
features), and optimize these for a while (also using cyclical learning
rates).

We then store these embeddings to disk so that they can be loaded later.

"""

import numpy as np
import pandas as pd

import pickle

from keras.models import Model, Sequential
from keras.layers import Input, Dense, Dropout, Activation
from keras.layers import Concatenate, Reshape, SpatialDropout1D
from keras.layers.embeddings import Embedding
from keras import regularizers

import pickle

print("Loading data...")

d  = load_training_data()
dval = load_validation_data()
dtest = load_test_data()

print("Loading features...")

cat_features = load_categorical_features()
num_features = load_numeric_features()

target = 'profit'

X=d.drop([target], axis=1)
y = np.log(1.0+d[target].astype(float))

Xval = dval.drop([target], axis=1)
yval = np.log(1.0+dval[target].astype(float))

Xtest = dtest.drop([target], axis=1)
ytest = np.log(1.0+dtest[target].astype(float))


inputs = []
embeddings = []

for col in cat_features:
    # find the cardinality of each categorical column:
    cardinality = int(np.ceil(d[col].nunique() + 2))
    # set the embedding dimension:
    # at least 2, at most 50, otherwise cardinality//2
    embedding_dim = max(min((cardinality)//2, 50),2)
    print(f'{col}: cardinality : {cardinality} and embedding dim: {embedding_dim}')
    col_inputs = Input(shape=(1,))
    # Specify the embedding
    embedding = Embedding(cardinality, embedding_dim,
                          input_length=1, name=col+"_embed")(col_inputs)
    # Add a but of dropout to the embedding layers to regularize:
    embedding = SpatialDropout1D(0.1)(embedding)
    # Flatten out the embeddings:
    embedding = Reshape(target_shape=(embedding_dim,))(embedding)
    # Add the input shape to inputs
    inputs.append(col_inputs)
    # add the embeddings to the embeddings layer
    embeddings.append(embedding)

# paste all the embeddings together
x = Concatenate()(embeddings)
# Add some general NN layers with dropout.
x = Dense(1024, activation='relu')(x)
x = Dropout(0.3)(x)
x = Dense(512, activation='relu')(x)
outputs = Dense(1)(x)

# Specify and compile the model:
embed_model = Model(inputs=inputs, outputs=outputs)
embed_model.compile(loss= "mean_squared_error",
                    optimizer="adam",
                    metrics=["mean_squared_error"])


def embedding_preproc(X_train, X_val, X_test, cat_cols, num_cols):
    """
    return lists with data for train, val and test set.

    Only categorical data, no numeric. (as we are just building the
    categorical embeddings)
    """
    input_list_train = []
    input_list_val = []
    input_list_test = []


    for c in cat_cols:
        input_list_train.append(X_train[c].values)
        input_list_val.append(X_val[c].values)
        input_list_test.append(X_test[c].values)

    return input_list_train, input_list_val, input_list_test

# get the lists of data to feed into the Keras model:
X_embed_train, X_embed_val, X_embed_test = embedding_preproc(
                                                X, Xval, Xtest,
                                                cat_features, num_features)

# Fit the model
embed_history = embed_model.fit(X_embed_train,
                                y.values,
                                validation_data = (X_embed_val, yval.values),
                                batch_size=1024,
                                epochs=15)


# Now copy the trained embeddings to a dict, and save the dict to disk.
embedding_dict = {}

for cat_col in cat_features:
    embedding_dict[cat_col] = embed_model.get_layer(cat_col + '_embed')\
                                         .get_weights()[0]

    print(f'{cat_col} dim: {len(embedding_dict[cat_col][0])}' )

pickle.dump(embedding_dict, open(str('embedding.dict'), "wb"))
