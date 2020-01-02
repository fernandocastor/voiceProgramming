# DEEP LEARNING WITH PYTHON by François Cholet.

## BASIC CONCEPTS

### Artificial Intelligence
Artificial intelligence is the area of Computer Science that attempts to automate intellectual tasks normally performed by humans. **Symbolic AI** involves the use of rule-based approaches for AI where developers attempt to hand-code rules to deal with every possible situation. Good for logical problems, such as playing chess. Not so good for more complex and fuzzy problems, such as image classification. 

### Machine learning

Machine learning approaches leverage data (inputs and expected outcomes) and produce rules. The latter can then be applied to new data to produce answers with which they have not been trained, original answers. To apply machine learning to a new problem, we need three things: (i) input data points; (ii) correspondingly matching output data points; (iii) an adjudication mechanism to determine whether the current output matches the expected output or not. Using the distance between actual and expected outputs and adjusting the way the "algorithm" (not really an algorithm, but a set of numeric coefficients) works is called learning. A machine learning approach **transforms** input data into meaningful outputs. The outputs become more meaningful as the approach is subjected to more (input, output) pairs and adjusts itself (learns). The adjustments that such an approach performs modify the way the input is **represented**. "Learning" pertains to the process of automatically searching for better representations. 

### Deep learning 

In the past, neural networks usually involved three layers, the input, output, and hidden layers. Neural networks modeled in this manner are still considered shallow machine learning approaches. Deep learning models use potentially many more layers. They leverage successive layers of increasingly meaningful representations. The number of layers is called the depth of the model. Deep learning models may leverage hundreds of even thousands of layers. Each layer consists of a sequence of numbers that are used to transform the input called **weights**. The aforementioned learning happens by adjusting the weights so as to more precisely transform inputs into expected outputs. The **loss function**, or **objective function**, calculates a distance between the actual and expected outputs. This loss score is then used to adjust the weights, by leveraging the backpropagation algorithm. Deep learning leverages neural networks. There are different kinds of deep neural network, more useful to solve different kinds of problems. For computer vision, *convolutional neural networks* are typically use. On the other hand, *recurrent neural networks* are good fits for timeseries data.

### Other Machine Learning approaches

Just mentioning the names of some "shallow" machine learning approaches. **Naive bayes** and **logistic regression** are basic classification approaches based on probabilistic modeling. **Support Vector Machines** is a kernel-based method that aims to find a decision boundary between two sets of points belonging to two different categories. Determining this boundary in a multidimensional space for many points is computationally intractable. The kernel function however is capable of calculating the distance between any two points. This can be done efficiently. This function must be created manually. The learning in SVMs is related to the separation hyperplane; the decision boundary between the sets of points is learned. This method does not scale up well to large datasets. **Decision trees** are flowchart-like and work for both classification and prediction. Each non-terminal in the tree represents a question and that is the part that is learned, e.g., is coefficient 2 in the data greater than 3.5? **Random forests** are a more powerful version of decision trees that involve a large number of specialized trees whose results are combined. **Gradient boosting machines** combine weaker prediction models, such as decision tres, but also leverage **gradient boosting**, which consists of iteratively training new models that specialize in addressing the shortcomings of the previous ones. Useful for structured data.

### Main advantages of Deep Learning

The first advantage is that, for a number of problems, *their performance (in terms of accuracy) is very good*. Second, it completely automates feature engineering. According to the book, the principle of using multiple layers does not work well with shallow learning approaches. From what I understood, in Deep Learning all the layers learn together. Shallow approaches in combination, on the other hand, would learn greedily, each layer independently. The second advantage is that deep learning is scalable because it is highly amenable to parallelization. Since deep learning involves highly parallelizable numerical computation, in particular matrix operations, GPUs are very important for deep learning. NVIDIA's TITAN and Tesla series are examples of very powerful GPUs. The third advantage is that previously trained deep learning models can be complemented by providing additional data. It is also possible to reuse previously trained models. A shortcoming of deep learning in general is that it usually requires a lot of data. 


## MATHEMATICAL BLOCKS OF NEURAL NETWORKS

**Keras** is a Python library for Deep Learning. 

**MNIST** is a dataset of 60,000 training images, plus 10,000 testing ones, of handwritten digits that must be classified in terms of ten categories (0-9). 

**10-way softmax layer** is a NN layer that will output an array of 10 probability scores (which add up to 1). This is what is used in neural networks that address the problem of correctly classifying MNIST data.

The **fit** verb often employed in this context refers to the process of adjusting the model to its training data. An **overfitted** model is one that is biased by the data in such a way that it works very well with that specific data but not with other inputs. More specifically, if we break down the (labeled) data in a training and a validation set, an overfitting means that the loss for the training set will go down whereas the loss for the validation set will go up. When a model is overfit, it is learning patterns from the data that are **irrelevant** for the task at hand (they do not help the network to work on never-seen-before inputs) but **optimize** its performance for the training data. 

A **tensor** is a container for data, almost always numerical data. A matrix is a 2D tensor. A tensor is an *N-dimensional matrix*, basically. For tensors, dimensions are often called axes (sing., axis). The num of axes of a tensor is its **rank**. A scalar is a 0D tensor. Examples include Python's float32 and float64. A 5 element vector is a 5-dimensional vector, but a 1D or rank 1 tensor (because vectors are inherently unidimensional, e.g., only have one axis). In deep learning, tensors are usually 0D to 4D, though 5D may be used for video. 

A tensor is usually represented as a numpy array. The number of axes can be obtained through the `ndim` property. It is also possible to discover the `shape` of the tensor. A vector has a shape such as `(5,)`, whereas a scalar has `()`. The `dtype` returns the type of the elements of a numpy array. Unlike in a Python list, elements in a numpy array all have the same type. Notation for obtaining slices in a numpy array is slightly different from lists: `train_images[10:100]` is equivalent to `train_images[10:100, :, :]`, considering that `train_images` is a rank 3 (three-dimensional) tensor. 

Deep learning models process data in batches, sequences of samples of the input data. In general, the first axis of a data tensor is called the **"samples axis"**. For example, the MNIST data set of digit images  has three axes: the samples axis and the remaining two, which represent the image data (28x28 images). The batches correspond to sequences of elements along the first axis. Batches are themselves tensors of the same dimension as the tensors they originate from. The first axis (axis 0) of a batch tensor is called the batch axis. 

Examples from the real world: 

- Vector data are 2-D tensors (samples, features)
- Images (colored) are 4-D tensors (samples, height, width, color channels)
- Videos are 5-D tensors (they are collections of images)

An example is encoding a single tweet. It could be encoded as a 2-D tensor where characters are encoded in a 128 bit vector, all zeros except for the bit corresponding to the char, and the tweet itselft consists of 280 such vectors corresponding to the characters. Such a tensor would have shape (280, 128). 

Tensor operations are what transforms the input. Examples include addition, multiplication, dot product, filtering out values lower than zero (and replacing them by zero), among others. The latter is called **relu** (Rectified Linear Unit) and is the most common of these tensor operations. Tensor operations are, in general, just linear algebra operations. With numpy it is possible to perform array addition and element-wise multiplication in a very optimized way using the `+` and `*` operators. There's also an element-wise maximum function. The `copy()` function creates a copy of an array independently of the number of axes. 

The element-wise operations can be performed between tensors of different dimensions. In this case, axes are added to the smaller tensor and the smaller has its contents repeated along these new axes so as to match the shape of the larger tensor. When we do this, we say that the smaller tensor was **broadcasted** to match the shape of the larger one. In general, it is possible to perform broadcast of a tensor *y* to match the shape of *x* if their dimensions are as follows: 

- *x* has shape *(a, b, ..., n, n+1, ..., m)*, and
- *y* has shape *(n, n+1, ..., m)*
  
In this scenario the broadcast of y happens for axes *a* through *n-1*. It is performed automatically when applying some element-wise operations to tensors. 

In Tensorflow, linear algebra operations are performed by BLAS, *Basic Linear Algebra Subprograms*, low-level, very efficient tensor manipulation routines. In numpy, `z=x+y`, where `x` and `y` are numpy arrays, will perform element-wise addition very efficiently. 

**Tensor product** or **dot or cross product** is a very important tensor operation that is **not element-wise**. In numpy, it is expressed as

```python
z = np.dot(xy) # only possible if x.shape[1] == z.shape[0], assuming ndim == 2
               # for both, and the resulting matrix has shape 
               # (x.shape[0], y.shape[1])
```

The dot product is analogous to matrix multiplication, but generalized to an arbitrary number of axes. In the general case, the relationship between shapes is akin to relational joins: 

```
(a,b,c,d).(d,) -> (a,b,c)
(a,b,c,d).(d,e) -> (a,b,c,e)
```

Tensors may be **reshaped** by rearranging coefficients to match a target shape. The coefficients are presered, but the shape changes. For examle:

```python
x = np.array([0.,1.], [2., 3.], [4., 5.]) # x.shape = 3,2
x.reshape((6,1))
x                                         # array([0.], [1.], [2.], [3.], [4.], [5.])
```

A common case of reshaping is transposition (`np.transpose()`). If `x.shape()==(x,y)`, `np.transpose(x).shape() == (y,x)`.

A neural network layer includes tensors of its own. These are shown in an operation such as `output = relu(dot(w, input) + b)`. In this example, `w`and `b` are **weights**, **coefficients**, or **trainable parameters** of the layer (for this specific case, known as "kernel" and "bias", respectively). They are adjusted through training, which happens gradually. The **training process** goes in four phases:

  1. obtain a batch of training samples `x` and expected results `y`; 
  2. provide the samples as input to the network and obtain the predicted values; 
  3. compare the latter with `y`, compute the loss (some measure of the distance between `y` and the predicted value); 
  4. adjust the weights in a way that slightly reduces the loss for this batch

**Gradients** and **Stochastic Gradient Descent** (which is a gradient descent approach that performs sampling so as to be more efficient)
**Backpropagation**
---

**Momentum** is an optimization method to improve convergence speed during training and avoid local minimums. The basic idea of momentum is that coefficients (weights) of the network should be updated based not only on the current gradient but also on how the values have been updated previously. 

The optimizer directly influences how the network will leverage stochastic gradient descent. It specifies how the gradient of the loss will be used to update the network parameters (coefficients). The loss is the quantity we're trying to minimize during training, the measure of the distance between the expected and estimated output of the network (the value into which it transforms the input). 


## GETTING STARTED WITH NEURAL NETWORKS

The most common uses of neural networks are binary classification, multiclass classification, and scalar classification. **Layers** are the basic building blocks of neural networks. They are usually stateful. Different types of layers are better fit to different tasks. Processing simple vector data in 2D tensors (samples, features) is usually the task of densely of fully connected layers (the `Dense` class in Keras). Timeseries data in 3D tensors is usually processed by recurrent layers (LSTM). Image data stored in 4D tensors is usually processed by 2D convolution layers (`Conv2D`). A deep learning model is constructed by connecting layers. Connnected layers must be compatible: the output of one must have the same shape as the input of the next one in the model. In Keras, models are automatically built so that developers do not need to worry about this when connecting multiple layers: 

```python
from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Dense(32, input_shape=(784,))) # 2D tensor. Axis 0, the batch 
                                       # dimension is left unspecified.
model.add(layers.Dense(32))            # does not specify the shape of the input
```

A deep learning model is a layer DAG. Examples of common topologies include linear stack. two-branch networks, multihead networks, and inception blocks. Linear models, e.g., a stack, are more common than any other topology. The network topology defines a hypothesis space. The topology constrains the tensor operations that may be applied to the input data and thusly the space of solutions that may be identified as useful representations of the input data. 

The **loss function** calculates how far the output is from the expected input. This distance is a number that the **optimizer** aims to minimize by adjusting the weights of the network, the **learning** process. The optimizer implements some variant of stochastic gradient descent (SGD). A neural network may have multiple outputs, each one with a different loss function. The learning process, however, more specifically, the SGD process, must be based on a single scalar. **Not sure why**, but I assume that this is necessary because the network is updated as a whole, not as isolated parts. When there are multiple outputs they are combined, possibly by averaging. The loss function is also called **objective function**. There are well-established objective functions for well-known problems, e.g., **binary crossentropy** for two class classification, or **mean squared error** for regression problems. Crossentropy measures the distance between two probability distributions. 

**Keras** is a model-level library for deep learning. It may use Tensorflow, Theano, or Microsoft's Cognitive Toolkit (CNTK) as backends. Multiple datasets come packaged with keras (module `keras.datasets`), including MNIST and IMDB datasets. In the IMDB dataset, training data items training data items are lists of word indices. The actual words to which these indices refer do not matter for the specific case of classifying movie reviews as positive or negative. It is possible to decode it, though. The dataset does keep a word index. The first three indices (0, 1, and 2) are reserved for padding, start of seq., and unknown. 

Neural network layers do not work with lists. They have to be made into tensors, either by passing them, after being padded so that all of them have the same length, to an embedding layer (if they are integer??just ints??? lists) or by **one-hot encoding** them, e.g., `[7, 42]` becomes an N-dimensional vector where N is the overall number of words (assuming we're working with text), and all the values of that vector are 0, except for indices 7 and 42. In the latter case, the first layer could be a `Dense` layer. 

-------
**Quick side note about numpy arrays**: 

In the code snippet 

```python
results[i, seq] = 1
```

if results is a numpy array and `seq` is a list, this statement will assign 1 to every position `[i,j] of results where `j` is an **element** of seq (an *element*, not an index), assuming that seq contains only integers. 

-------

The labels (for the expected results for the given inputs) also have to be vectorized, .i.e, it is necessary to build a numpy array out of each list, specifying the element types. 

When designing networks that use dense layers, two important parameters that must be determined are the **number of layers** of the network and the **number of hidden units** for each layer. The latter determine the number of dimensions into which the data will be projected. The greater the number of hidden units, the more complex the representations that the network will be capable of learning. The downside to having higher dimensionality in terms of hidden units is that training becomes more resource-intensive and the network becomes more likely to learn unwanted patterns. 

**Activation functions** are what makes the transformations applied by a neural network layer non-linear. **Rectified Linear Unit (relu)** is an example of such a function, the most popular one. Working with non-linear transformations enriches the hypothesis space which would otherwise be built based solely on linear transformations. For example, for a relu layer with 16 hidden units, without an activation function the hypothesis space would consist of all the possible linear transformations of the input vector (assuming it is a vector) into a 16-dimensional space. If all the transformations are linear, there is no need to have a deep stack of layers because such a stack would still implement a linear operation (and can be composed into a single operation). Activation functions can also be called **``non-linearity functions``**. 

After having trained the model, it is possible to use it on new data using method ``predict()`` from the model object. Since we're using a sigmoid function to produce error probabilities (in the example from the book -- a layer that uses **sigmoid activation** outputs a probability), this function will output an array of probabilities, one for each input element. Section 3.4.6 suggests some further experiments (changes to the NN architecture) to show that the one we employed is reasonable.

To classify data when there are multiple classes, we can use **single- or multi-label multiclass classification**, depending on whether each data point can be included in a single class or in multiple ones. 

-----
**Reuters dataset**: short newswires and their topics. 46 topics, each one applies to at least 10 data points. Part of keras (from `keras.datasets import reuters`). Similar to the imdb dataset, except for being multiclass. Therefore, instead of 0s and 1s, the labels range between 0 and 45. To encode labels using one-hot encoding (aka *"categorical encoding"*), we can use code similar to page 69. Alternatively, it is possible to use the `to_categorical()` function from `keras.utils.np_utils`. 

A 16-dimensional intermediate layer may be too constraining to learn to classify into 46 separate categories. 64 hidden units, instead of 16, may be more appropriate. In addition, for the last layer (a `Dense` layer of size 46, the number of classes), the activation function will be **softmax**, because it will output a probability distribution vector over the 46 different categories, i.e., each element of the 46-dimensional vector that will be output is a value between 0 and 1 but the overall sum of these values is 1. 

The **number of epochs** during which a neural network should be trained is very important. Stopping the training at peak accuracy (based on validation data) can avoid overfitting. 

If, instead of using one-hot encoding for the labels, we just made them into an integer tensor (creating a numpy array), the difference would be that we'd use `sparse_categorical_crossentropy` as the loss function, since `categorical_crossentropy` (the one we used before -- have to look at the examples in the book) expects categorical data. 

-----
**Boston Housing dataset** (`from keras.datasets import boston_housing`): It includes 404 training samples and 102 test samples. Each feature in the input data has a different scale (some are proportions between 0 and 1, others have values between 0 and 12, etc.). Each data point has 13 features. Features in different scales are typically normalized by subtracting the mean and dividing by the standard deviation (so that every feature is expressing in # of standard devitations). This is performed for every feature. **Test data also needs to be normalized**. However, it is not used in the calculation of the mean and the standard deviation. In fact, one should **never use any value computed based on testing data for anything other than testing**!

Machine learning can also be employed to predict continuous values, instead of discrete labels. This kind of problem is called **regression**. An estimate of the duration of a task, based on previous executions of said task, is an example of a regression problem. BTW, **logistic regression** is not a regression in this sense because it actually classifies (produces categorical output). 

In this scalar regression scenario, where we're trying to predict a single value, the last layer does not use an activation function, since it will simply output a linear value. It  is still necessary to specify that it will output a single value, though. 

**Question**: Why is the network for Boston Housing small? Looks like the one for the Reuters news. *Answer*: The dataset is too small. If it is too big, it is guaranteed to overfit due to memorizing each and every possible pattern from the data. It will have no capability to generalize. 

For a scenario where a neural network outputs a linear value, it makes sense to use mean squared error as the objective function ("`mse`"). Furthermore, mean absolute error ("`mae`") makes more sense to use as the metric to monitor, since using accuracy for values that are unbounded does not make sense (because the values are potentially very different and there is no frame of reference to indicate, for example, an error percentage). For this example, where the outputs are expressed in thousands of dollars, a `mae` of 0.5 would mean that the predictions are off by $500 on average. 

Validation must be a bit different for this example  because the dataset is so small. In this scenario, it is typical to use **k-fold cross validation**. In summary, this means that the data is broken down into *k* partitions (usually 4 or 5, though **software engineering people usually employ 10, IDK why**), create *k* copies of the model and train each one with a different group of *k-1* partitions, using the remaining partition to validate. In this case, the validation score will be the average of the *k* validations.


-------
**Quick side note about element-wise operations and numpy arrays**: 

If an element-wise operation such as addition or subtraction is applied to a numpy array and a scalar value, it will compute that operation using each element of the array as its first operand and the scalar value as the second one. For example:

```python
mean = train_data.mean(axis=0)
train_data -= mean # train data is a tensor, mean is a scalar 
```
-------

In the first snippet of page 88, the `axis` parameter must be explicitly passed (0) to the `concatenate()` function because otherwise the resulting array will be flattened, i.e., its elements will not themselves be arrays. For example:

```python
a = np.array([1,2], [3,4])
b = np.array([[5,6]])
np.concatenate((a,b), axis=0)     # array([[1,2], [3,4], [5,6]])
np.concatenate((a,b), axis=None)  # array([1,2,3,4,5,6])
```


## FUNDAMENTALS OF MACHINE LEARNING

There are four broad categories of machine learning algorithms: (i) supervised learning, (ii) unsupervised learning, (iii) self-supervised learning (iv) reinforcement learning. 

Every example we've seen fits into the first category. This approach consists of learning to map input data to known targets given a set of examples previously annotated by humans. Most deep learning applications fit into this. Most problems tackled with this approach consist of either classification or regression. Other examples include sequence generation, e.g., predicting a caption to a picture, syntax tree prediction, object detection, and image segmentation. 

Reviewing some terminology for classification and regression (pages 95 and 96):

- **sample or input**: one data point fed to the model
- **target**: what the model should have predicted
- **ground truth or annotations**: all the targets for a dataset
- **vector regression**: a task where the target is a set of continuous values, e.g., a continuous vector
- **mini-batch or batch**: a small set of samples that are processed simultaneously by the model (possibly in parallel by a GPU). During training, a mini-batch is used to compute a single gradient-descent update to the weights of the model. 

**Why do we need three dataset partitions, training, validation, and testing?** In summary, because the validation data is what we use to tune the hyperparameters of the model, e.g., its number of layers or their size. "Hyperparameters" because "parameters" are the network's weights. This tuning can lead to overfitting to the validation set (instead of just the training set). When we train a model, we may adjust its hyperparameters so as to improve results on the validation set. Every time such an adjustment is performed, some information about the validation set **leaks** into the model. If this kind of adjustment is performed repeatedly, these information leaks may lead to a model that is overfitted to the validation data. Because of this phenomenon,  a model must be tested on a dataset with which it had no prior contact, not even indirectly. After all, the entire goal of training and validation is to produce a model that can be applied to new data, e.g., a generalizable model.

Splitting data among training, validation, and testing data may be difficult when little data is avaliable. In this case, besides using **simple hold-out validation**, one can employ **k-fold cross validation** and **iterated k-fold validation with shuffling**. The later consists of applying the former multiple times. re-shuffling the data every time before splitting again into *K* partitions. The final score is the average of the scores of every run. This provides greater reliability of evaluation, at the cost of training and evaluating *P* * *K* models, where *P* is the number of iterations. It is important to mention that if the dataset corresponds to some kind of timeseries and the aim is to predict the future, the data should not be shuffled. Furthermore, it is necessary to ensure that all the data in the testing set is posterior to the data in the training set. 

-------
**Training a model on its training+validation data** after the model has been tuned is a common practice. It is even intuitive: By then, parts of the validation data may have leaked into the model anyway. 

On the other hand, it is critical to ensure that, in scenarios where there is redundancy in the dataset, training and test data are disjoint. 

-------

Although deep neural networks require less feature engineering than other ML approaches, it is still useful to **preprocess data for neural networks**. The main preprocessing steps are **vectorization**, **normalization**, **handling of missing values**, and **feature extraction**. 

Normalization is important because, if data is too heterogeneous (some features range between 0 and 1 whereas others range between 10 and 200), this will trigger large gradient updates, which will hinder network convergence. Ideally, data should (i) take small values, (ii) be homogeneous (in terms of range of values, e.g., normalized in terms of std. devs. from the mean). General rule of thumb: normalize each feature independently to have a mean of 0 and a standard deviation of 1. In numpy: 

```python
x -= x.mean(axis=0)  # x is a 2D data matrix of shape
x /= x.std(axis=0)   # (samples, features)
```

For missing values, use 0 (unless 0 already means something). If they are expected but the samples do not have missing values, it is possible to generate them. This will teach the network to ignore them. 

**Feature engineering** is more important for shallow machine learning than for DL because shallow approaches do not have a rich enough hypothesis space to learn useful features by themselves. Feature engineering is still useful in DL, however, because it can reduce resource usage and the number of samples required for training. 

**Regularization approaches** aim to reduce model overfitting. The ideal approach to deal with overfitting is simply to train the model on more data. However, this is often not possible. In this scenario, regularization techniques can help mitigate the problem. The first approach one can leverage is to **Reduce the Model's Capacity**. The capacity of a model is its number of learnable parameters. A greater model capacity implies a greater memorization capacity. Reducing capacity may stop the model from learning useless features that would only optimize its performance for the training data but not to never-seen-before data. In deep learning, the challenge is often to generalize and not just to fit. Another approach to tackle overfitting is **Weight Regularization**. This approach forces weights to "take small values". Keras supports this directly in the construction of the model (Listing 4.6, page 108). **Adding Dropout** is a common and very powerful regularization technique. It consists of determining a **dropout rate** (usually between 0.2 and 0.5), the percentage of the elements of the output vector of a layer that will be made 0 at training time. At testing time, this rate is applied as a reduction in the values of the outputs produced by that layer. Alternatively, these two operations may be combined at training time and the output is left unchanged at testing time. The rationale for using this approach is that introducing noise in the output helps break coincidental, happenstance patterns that are not significant. In keras, it is possible to add a `Dropout` layer:

```python
model.add(layers.Dropout(0.5))    # 50% dropout rate
```

-------

### The Universal Blueprint to Attacking Machine Learning Problems**

Two important hypothesis must hold for machine learning to be used: 

- Outputs can be predicted given the inputs
- Available data is sufficiently informative to learn the relationship between inputs and outputs

The steps:

1. Define the problem and assemble a dataset
2. Choose a measure of success, e.g., accuracy, precision and recall, ROC AUC (Receiving Operation Characteristic Curve -- Area Under Curve). The choice of metrics depends on the problem.
3. Decide on an evaluation protocol, e.g., simple hold-out, k-fold cross val., iterated k-fold shuffling.
4. Prepare the data, e.g., normalize, encode, do some engineering if useful. 
5. Develop a model that fares better than the baseline, e.g., for classification something that outperforms a random classifier. This means that the model has statistical power. Hyperparameters are set up, a loss function is determined, activation functions for the layers are determined (or maybe the last one will not use one in case it should produce a scalar), and an optimizer is picked, e.g., `rmsprop`. Table 4.1 of page 114 provides an overview. 
6. Make the model overfit
7. Regularize the model and tune hyperparameters

-------


## Chapter 5

About transfer learning and data augmentation and why reusing a model in a more direct, OTS way, without incorporating it into our own custom model, does not provide support for data augmentation approaches. 

https://stats.stackexchange.com/questions/331162/feature-extraction-freezing-convolutional-base-vs-training-on-extracted-featur/366465?newreg=4f0695a759ea4446ad76161b0e495093

