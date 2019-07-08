
# Deep Learning Entity Embedding model in Keras
by: Oege Dijk

Neural Network models are almost always better for unstructured data (e.g.
  image data).  However for structured data, they often still underperform
  tree based models (random forrests, boosted trees, etc) they often also don't
  play as nice with categorical variables as tree models do.

However an exciting new methodology to work with categorical data is
entitiy embeddings. These are similar to word embeddings to those who are
familiar with NLP models. Check out
 [lesson 11 of the fast.ai course Introduction to Machine Learning for Coders](https://course.fast.ai/lessonsml1/lesson11.html)
 for a nice overview lecture. Or [this article if you prefer to read instead of watching video](https://towardsdatascience.com/decoded-entity-embeddings-of-categorical-variables-in-neural-networks-1d2468311635)

Embeddings are basically a way of replacing each instance of a categorical
variable by a vector of a particular length (rule of thumb is
  `len = min(cardinality/2, 50)`). The vector is initialized randomly just like
  any other layer in a neural network , and then updated through
gradient descent to find the values that minimize the loss function.

The result is that these vectors start representing *meaning*. If you train
such entity embeddings to represent countries for example, it could learn
that certain countries are similar along a certain dimension that turns out
to be 'Spanish speaking', or perhaps 'European', 'Buddhist', etc. Whatever
dimensions help the model in minimizing the loss function will get encoded into
these embeddings.

## Embeddings in Keras

There is not a whole lot of sample code for entity embeddings out there,
so here I share one implementation in Keras. This implementation also shows
how you can save the embeddings to disk, and then later load them into
another model.

We first train a model only using these embeddings (thus excluding
numerical features), and then save these embeddings to disk (see `build_embeddings.py`).

We can then load these embeddings, and then replace the categorical
  variables with these embeddings, and retrain the model along with
  the numerical features. (see `build_embed_model.py`)
