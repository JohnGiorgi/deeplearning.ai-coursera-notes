# Introduction

### Resources

- [Notation cheetsheet](https://d3c33hcgiwev3.cloudfront.net/_106ac679d8102f2bee614cc67e9e5212_deep-learning-notation.pdf?Expires=1514764800&Signature=bnnZZMJUAG2PZPWezLLN6EeKjihlTdaVPo1fqHDdsPmXkLDyjVG-fBtstgOCIcFkKd8OGx845pIKDITTFGm0sMA1eGo4lAIqP7Btffy5VGBRwasKW3WCGGkP-dmq0Vw7Y83ezax4wQCzzYB6iPevY8QniePzg-iq~O5a9hJ4TRk_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A) (I recommend printing this out and sticking it on the wall where you work!)
- Check links at end of all programming assignments, these are good resources.

## What is a neural network?

### Supervised Learning with Neural Networks

In supervised learning, you have some input \\(x\\) and some output \\(y\\) . The goal is to learn a mapping \\(x \rightarrow y\\) .

Possibly, the single most lucrative (but not the most inspiring) application of deep learning today is online advertising. Using information about the ad combined with information about the user as input, neural networks have gotten very good at predicting whether or not you click on an ad. Because the ability to show you ads that you're more likely to click on has a _direct impact on the bottom line of some of the very large online advertising companies_.

Here are some more areas in which deep learning has had a huge impact:

- __Computer vision__ the recognition and classification of objects in photos and videos.
- __Speech Recognition__ converting speech in audio files into transcribed text.
- __Machine translation__ translating one natural language to another.
- __Autonomous driving__

![](https://s19.postimg.org/wdqnmh0o3/supervised_learning.png.png)

A lot of the value generation from using neural networks have come from intelligently choosing our \\(x\\) and \\(y\\) and learning a mapping.

We tend to use different architectures for different types of data. For example, __convolutional neural networks__ (CNNs) are very common for _image data_, while __recurrent neural networks__ (RNNs) are very common for _sequence data_ (such as text). Some data, such as radar data from autonomous vehicles, don't neatly fit into any particularly category and so we typical use a complex/hybrid network architecture.

### Structured vs. Unstructured Data

You can think of __structured data__ as essentially meaning _databases of data_. It is data that is highly _structured_, typically with multiple, well-defined attributes for each piece of data. For example, in housing price prediction, you might have a database where the columns tells you the size and the number of bedrooms. Or in predicting whether or not a user will click on an ad, you might have information about the user, such as the age, some information about the ad, and then labels why that you're trying to predict.

In contrast, __unstructured data__ refers to things like audio, raw audio,
or images where you might want to recognize what's in the image or text. Here the features might be the pixel values in an image or the individual words in a piece of text. Historically, it has been much harder for computers to make sense of unstructured data compared to structured data. In contrast the human race has evolved to be very good at understanding audio cues as well as images. _People are really good at interpreting unstructured data_. And so one of the most exciting things about the rise of neural networks is that, thanks to deep learning, thanks to neural networks, computers are now much better at interpreting unstructured data as well compared to just a few years ago. And this creates opportunities for many new exciting applications that use speech recognition, image recognition, natural language processing on text.

Because people have a natural empathy to understanding unstructured data, you might hear about neural network successes on unstructured data more in the media because it's just cool when the neural network recognizes a cat. We all like that, and we all know what that means. But it turns out that a lot of short term economic value that neural networks are creating has also been on structured data, such as much better advertising systems, much better profit recommendations, and just a much better ability to process the giant databases that many companies have to make accurate predictions from them.

 ![](https://s19.postimg.org/66pgpyd37/unstructured_vs_structured_data.png.png)

## Why is Deep Learning taking off?

_If the basic technical details surrounding deep learning have been around for decades, why are they just taking off now?_

First and foremost, the massive amount of (labeled) data we have been generating for the past couple of decades (in part because of the 'digitization' of our society).

It turns out, that large, complex neural networks can take advantage of these huge data stores. Thus, we often say _scale_ has been driving progress with deep learning, where scale means the size of the data, the size/complexity of the neural network, and the growth in computation.

The interplay between these 'scales' is apparent when you consider that many of the algorithmic advances of neural networks have come from making them more computational efficient.  

#### Algorithmic Advances: ReLu

One of the huge breakthroughs in neural networks has been the seemingly simple switch from the __sigmoid__ activation function to the __rectified linear__ (ReLu) activation function.

One of the problems with using __sigmoid__ functions is that its gradients approach 0 as input to the sigmoid function approaches and \\(+\infty\\) and \\(-\infty\\) . In this case, the updates to the parameters become very small and our learning slows dramatically.

With ReLu units, our gradient is equal to \\(1\\) for all positive inputs. This makes learning with gradient descent much faster. See [here](https://www.wikiwand.com/en/Rectifier_(neural_networks)) for more information on ReLu's.

![](https://upload.wikimedia.org/wikipedia/en/thumb/6/6c/Rectifier_and_softplus_functions.svg/640px-Rectifier_and_softplus_functions.svg.png?1514655339839.png)

#### Scale Advances

With smaller training sets, the relative ordering of the algorithms is actually not very well defined so if you don't have a lot of training data it is often up to your skill at hand engineering features that determines the
performance. For small training sets, it's quite possible that if someone training an SVM is more motivated to hand engineer features they will outperform a powerful neural network architecture.

 ![](https://s19.postimg.org/6i6x39jer/scale.png.png)

However, for very large training sets, _we consistently see large neural networks dominating the other approaches_.

# Week 1

## Binary Classification

First, some notation,

- \\(n\\) is the number of data attributes, or _features_
- \\(m\\) is the number of input examples in our dataset (sometimes we write \\(m_{train}, m_{test}\\) to be more explicit).
- our data is represented as input, output pairs, \\((x^{(1)},y^{(1)}), ..., (x^{(m)},y^{(m)})\\) where \\(x \in \mathbb R^n\\) , \\(y \in \{0,1\}\\)
- \\(X\\) is our design matrix, which is simply columns of our input vectors \\(x^{(i)}\\) , thus it has dimensions of \\(n\\) x \\(m\\) .
- \\(Y = [y^{(1)}, ..., y^{(m)}]\\) , and is thus a \\(1\\) x \\(m\\) matrix.

> Note, this is different from many other courses which represent the design matrix, \\(X\\) as rows of transposed input vectors and the output vector \\(Y\\) as a \\(m\\) x \\(1\\) column vector. The above convention turns out to be easier to implement.

When programming neural networks, implementation details become extremely important (_e.g_. vectorization in place of for loops).

We are going to introduce many of the key concepts of neural networks using __logistic regression__, as this will make them easier to understand. Logistic regression is an algorithm for __binary classification__. In binary classification, we have an input (_e.g_. an image) that we want to classifying as belonging to one of two classes.

### Logistic Regression (Crash course)

Given an input feature vector \\(x\\) (perhaps corresponding to an images flattened pixel data), we want \\(\hat y\\) , the probability of the input examples class, \\(\hat y = P(y=1 | x)\\)

> If \\(x\\) is a picture, we want the chance that this is a picture of a cat, \\(\hat y\\) .

The parameters of our model are \\(w \in \mathbb R^{n_x}\\) , \\(b \in \mathbb R\\) . Our output is \\(\hat y = \sigma(w^Tx + b)\\) were \\(\sigma\\) is the __sigmoid function__.

![sigmoid](https://upload.wikimedia.org/wikipedia/commons/thumb/8/88/Logistic-curve.svg/640px-Logistic-curve.svg.png.png)

The formula for the sigmoid function is given by: \\(\sigma(z) = \frac{1}{1 + e^{-z}}\\) where \\(z = w^Tx + b\\) . We notice a few things:

- If \\(z\\) is very large, \\(e^{-z}\\) will be close to \\(0\\) , and so \\(\sigma(z)\\) is very close to \\(1\\) .
- If \\(z\\) is very small, \\(e^{-z}\\) will grow very large, and so \\(\sigma(z)\\) is very close to \\(0\\) .

> It helps to look at the plot \\(y = e^{-x}\\)

Thus, logistic regression attempts to learn parameters which will classify images based on their probability of belonging to one class or the other. The classification decision is decided by applying the sigmoid function to \\(w^Tx + b\\) .

> Note, with neural networks, it is easier to keep the weights \\(w\\) and the biases \\(b\\) separate. Another notation involves adding an extra parameters (\\(w_0\\) which plays the role of the bias.

#### Loss function

Our prediction for a given example \\(x^{(i)}\\) is \\(\hat y^{(i)} = \sigma(w^Tx^{(i)} + b)\\) .

We chose __loss function__, \\(\ell(\hat y, y) = -(y \; log\; \hat y + (1-y) \;log(1-\hat y))\\) .

We note that:

- If \\(y=1\\) , then the loss function is \\( \ell(\hat y, y) = -log\; \hat y\\) . Thus, the loss approaches zero as \\(\hat y\\) approaches 1.
- If \\(y=0\\) , then the loss function is \\(\ell(\hat y, y) = -log\; (1 -\hat y)\\) . Thus, the loss approaches zero as \\(\hat y\\) approaches 0.

> Note, while \\(\ell_2\\) loss is taught in many courses and seems like an appropriate choice, it is non-convex and so we cannot use gradient descent to optimize it.

> An optional video is given further justifying the use of this loss function. Watch it and add notes here!

Note that the __loss function__ measures how well we are doing on a _single example_. We now define a __cost function__, which captures how well we are doing on the entire dataset:

\\(J(w,b) = \frac{1}{m}\sum^m_{i=1} \ell(\hat y^{(i)}, y^{(i)}) = - \frac{1}{m}\sum^m_{i=1}(y^{(i)} \; log\; \hat y^{(i)} + (1-y^{(i)}) \;log(1-\hat y^{(i)}))\\)

> Note that this notation is somewhat unique, typically the cost/loss functions are just interchangeable terms. However in this course, we will define the __loss function__ as computing the error for a single training example and the __cost function__ as the average of the loss functions of the entire training set.

### Gradient Descent

We want to find \\(w,b\\) which minimize \\(J(w,b)\\) . We can plot the __cost function__ with \\(w\\) and \\(b\\) as our horizontal axes:

![cost-surface](https://s19.postimg.org/pna6cuy0z/cost_surface.png.png)

> In practice, \\(w\\) typically has many more dimensions.

Thus, the cost function \\(J(w,b)\\) can be thought of as a surface, were the height of the surface above the horizontal axes is its value. We want to find the values of our parameters \\(w, b\\) at the lowest point of this surface, the point at which the average loss is at its minimum.

__Gradient Descent Algorithm__

Initialize \\(w,b\\) to some random values

> because this cost function is convex, it doesn't matter what values we use to initialize, \\(0\\) is usually chosen for logistic regression.

Repeat

1. \\(w := w - \alpha \frac{dJ(w)}{dw}\\)
2. \\(b := b - \alpha \frac{dJ(w)}{db}\\)

> \\(\alpha\\) is our learning rate, it controls how big a step we take on each iteration. Some notation, typically we use \\(\partial\\) to denote the partial derivative of a function with \\(2\\) or more variables, and \\(d\\) to denote the derivative of a function of only \\(1\\) variable.

![gradient-descent](https://s19.postimg.org/a1susyr8j/gradient_descent.png.png)

When implementing gradient descent in code, we will use the variable \\(dw\\) to represent \\(\frac{dJ(w, b)}{dw}\\) (this size of the step for \\(w\\) and \\(db\\) to represent \\(\frac{dJ(w, b)}{db}\\) (the size of the step for \\(b\\) .

### (ASIDE) Calculus Review

#### Intuition about derivatives

##### Linear Function Example

Take the function \\(f(a) = 3a\\). Then \\(f(a) = 6\\) when \\(a = 2\\) . If we were to give \\(a\\) a tiny nudge, say to \\(a = 2.001\\) , what happens to \\(f(a)\\) ?

![derivative](https://s19.postimg.org/wdqnmadgz/derivative.png.png)

Then \\(f(a) = 6.003\\) , but more importantly if we inspect the triangle formed by performing the nudge, we can get the slope of the function between \\(a\\) and \\(a + 0.001\\) as the \\(\frac{height}{width} = 3\\) .

Thus, the __derivative__ (or slope) of \\(f(a)\\) _w.r.t_ \\(a\\) is \\(3\\) . We say that \\(\frac{df(a)}{da} = 3\\) or \\(\frac{d}{da}f(a) = 3\\)

> Add my calculus notes here!
> Link to BlueBrown videos.

##### Non-Linear Function Example

Take the function \\(f(a) = a^2\\) . Then \\(f(a) = 4\\) when \\(a = 2\\) . If we were to give \\(a\\) a tiny nudge, say to \\(a = 2.001\\), what happens to \\(f(a)\\)?

![more-derivatives](https://s19.postimg.org/535ceh5g3/more_derivatives.png.png)

Then \\(f(a) = 4.004\\), but more importantly if we inspect the triangle formed by performing the nudge, we can get the slope of the function between \\(a\\) and \\(a + 0.001\\) as the \\(\frac{height}{width} = 4\\) .

In a similar way, we can perform this analysis for any point \\(a\\) on the plot, and we will see that slope of \\(f(a)\\) at some point \\(a\\) is equal to \\(2a\\) .

Thus, the __derivative__ (or slope) of \\(f(a)\\) _w.r.t_ \\(a\\) is \\(2a\\) . We say that \\(\frac{df(a)}{da} = 2a\\) or \\(\frac{d}{da}f(a) = 2a\\) .

### Computation Graph

A __computation graph__ organizes a series of computations into left-to-right and right-to-left passes. Lets build the intuition behind a computation graph.

Say we are trying to compute a function \\(J(a,b,c) = 3(a + bc)\\) . This computation of this function actually has three discrete steps:

- compute \\(u = bc\\)
- compute \\(v = a + u\\)
- compute J = \\(3v\\)

We can draw this computation in a graph:

![computation-graph](https://s19.postimg.org/qcsyp86ab/computation_graph.png.png)

The computation graph is useful when you have some variable or output variable that you want to optimize (\\(J\\) in this case, in logistic regression it would be our _cost function output_). A _forward pass_ through the graph is represented by _left-to-right_ arrows (as drawn above) and a _backwards pass_ is represented by _right-to-left_ arrows.

A backwards pass is a natural way to represent the computation of our derivatives.  

#### Derivatives with a computation graph

Lets take a look at our computation graph, and see how we can use it to compute the partial derivatives of \\(J\\) i.e., lets carry out backpropogation on this computation graph by hand.

> Informally, you can think of this as asking: "If we were to change the value of \\(v\\) slightly, how would \\(J\\) change?"

![clean-computation-graph](https://s19.postimg.org/q01kj10v7/clean_computation_graph.png.png)

First, we use our informal way of computing derivatives, and note that a small change to \\(v\\) results in a change to \\(J\\) of 3X that small change, and so \\(\frac{dJ}{dv} = 3\\) . This represents one step in our backward pass, the first step in backpropagation.

Now let's look at another example. What is \\(\frac{dJ}{da}\\(?

We compute the \\(\frac{dJ}{da}\\) from the second node in the computation graph by noting that a small change to a results ina change to \\(J\\) of 3X that small  change, and so \\(\frac{dJ}{da} = 3.\\) This represents our second step in our backpropagation.

One way to break this down is to say that by changing \\(a\\), we change \\(v\\), the magnitude of this change is \\(\frac{dv}{da}\\) . Through this change in \\(v\\), we also change \\(J\\), and the magnitude of the change is \\(\frac{dJ}{dv}\\) . To captue this more generally, we use the __chain rule__ from calculus, informally:

\\[\text{if } a \rightarrow v \rightarrow J \text{, then } \frac{dJ}{da} = \frac{dJ}{dv} \frac{dv}{da}\\]

> Here, just take \\(\rightarrow\\) to mean 'effects'. A formal definition of the chain rule can be found [here](https://www.wikiwand.com/en/Chain_rule).

The amount \\(J\\) changes when you when you nudge \\(a\\) is the product of the amount \\(J\\) changes when you nudge \\(v\\) multiplied by the amount \\(v\\) changes when you nudge \\(a\\) .

> __Implementation note__: When writing code to implement backpropagation, there is typical a single output variable you want to optimize, \\(dvar\\), (the value of the cost function). We will follow to notation of calling this variable \\(dvar\\) .

If we continue performing backpropagation steps, we can determine the individual contribution a change to the input variables has on the output variable. For example,

\\[\frac{dJ}{db} = \frac{dJ}{du} \frac{du}{db} = (3)(2) = 6\\]

The key take away from this video is that when computing derivatives to determine the contribution of input variables to change in an output variable, the most efficient way to do so is through a right to left pass through a computation graph. In particular, we'll first compute the derivative with respect to the output of the left-most node in a backward pass, which becomes useful for computing the derivative with respect to the next node and so forth. The __chain rule__ makes the computation of these derivatives tractable.

### Logistic Regression Gradient Descent

Logistic regression recap:

\\[z = w^Tx + b\\]
\\[\hat y = a = \sigma(z)\\]
\\[\ell(a,y) = -(ylog(a) + (1-y)log(1-a))\\]

> \\(\ell\\) is our loss for a single example, and \\(\hat y\\) are our predictions.

For this example, lets assume we have only two features: \\(x_1\\), \\(x_2\\) . Our computation graph is thus:

![logistic-regression-computation-graph](https://s19.postimg.org/jz3vlzgtv/computation_graph_logression.png.png)


Our goal is to modify the parameters to minimize the loss \\(\ell\\) . This translates to computing derivatives \\(w.r.t\\) the loss function. Following our generic example above, we can compute all the relevant derivatives using the chain rule. The first two passes are computed by the following derivatives:

1. \\(\frac{d\ell(a,y)}{da} = - \frac{y}{a} + \frac{1-y}{1-a}\\(
2. \\(\frac{d\ell(a,y)}{dz} = \frac{d\ell(a,y)}{da} \cdot \frac{da}{dz} =  a - y\\(

> Note: You should prove these to yourself.

> __Implementation note__, we use \\(dx\\) as a shorthand for \\(\frac{d\ell(\hat y,y)}{dx}\\) for some variable \\(x\\) when implementing this in code.


Recall that the final step is to determine the derivatives of the loss function \\(w.r.t\\) to the parameters.

- \\(\frac{d\ell(a,y)}{dw_1} = x_1 \cdot \frac{d\ell(a,y)}{dz}\\)
- \\(\frac{d\ell(a,y)}{dw_2} = x_2 \cdot \frac{d\ell(a,y)}{dz}\\)

One step of gradient descent would perform the updates:

- \\(w_1 := w_1 - \alpha \frac{d\ell(a,y)}{dw_1}\\)
- \\(w_2 := w_2 - \alpha \frac{d\ell(a,y)}{dw_2}\\)
- \\(b := b - \alpha \frac{d\ell(a,y)}{db}\\(

#### Extending to \\(m\\) examples

Lets first remind ourself of the logistic regression __cost__ function:

\\[J(w,b) = \frac{1}{m}\sum^m_{i=1} \ell(\hat y^{(i)}, y^{(i)}) = - \frac{1}{m}\sum^m_{i=1}(y^{(i)} \; log\; \hat y^{(i)} + (1-y^{(i)}) \;log(1-\hat y^{(i)}))\\]

Where,

\\[\hat y = a = \sigma(z) = \sigma(w^Tx^{(i)} + b)\\]

In the example above for a single training example, we showed that to perform a gradient step we first need to compute the derivatives \\(\frac{d\ell(a,y)}{dw_1}, \frac{d\ell(a,y)}{dw_2}, \frac{d\ell(a,y)}{db}\\) . For \\(m\\) examples, these are computed as follows:

- \\(\frac{\partial\ell(a,y)}{\partial dw_1} = \frac{1}{m}\sum^m_{i=1} \frac{\partial}{\partial w_1} \ell(\hat y^{(i)}, y^{(i)})\\)
- \\(\frac{\partial\ell(a,y)}{\partial w_2} = \frac{1}{m}\sum^m_{i=1} \frac{\partial}{\partial w_2} \ell(\hat y^{(i)}, y^{(i)})\\)
- \\(\frac{\partial\ell(a,y)}{\partial b} = \frac{1}{m}\sum^m_{i=1} \frac{\partial}{\partial b} \ell(\hat y^{(i)}, y^{(i)})\\)

We have already shown on the previous slide how to compute \\(\frac{\partial}{\partial w_1} \ell(\hat y^{(i)}, y^{(i)}), \frac{\partial}{\partial w_2} \ell(\hat y^{(i)}, y^{(i)})\\) and \\(\frac{\partial}{\partial b} \ell(\hat y^{(i)}, y^{(i)})\\) . Gradient descent for \\(m\\) examples essentially involves computing these derivatives for each input example \\(x^{(i)}\\) and averaging the result before performing the gradient step. Concretely, the pseudo-code for gradient descent on \\(m\\) examples of \\(n=2\\) features follows:

__ALGO__

Initialize \\(J=0; dw_1 = 0; dw_2 = 0; db = 0\\)

for \\(i=1\\) to \\(m\\):

- \\(z^{(i)} = w^Tx^{(i)}\\)
- \\(a^{(i)} = \sigma(z^{(i)})\\)
- \\(J \text{+= } -[y^{(i)}log(a^{(i)}) + (1-y^{(i)})log(1-a^{(i)})]\\)
- \\(dz^{(i)} = a^{(i)} - y^{(i)}\\)
- for \\(j = 1\\) to \\(n\\)
  - \\(dw_j \text{+= } x_j^{(i)}dz^{(i)}\\)
  - \\(dw_j \text{+= } x_j^{(i)}dz^{(i)}\\)
- \\(db \text{+= } dz^{(i)}\\)

\\(J = J/m;\; dw_1 \text{=/ } m;\; dw_2 \text{=/ }  m;\;b \text{=/ }  m\\)

In plain english, for each training example, we use the sigmoid function to compute its activation, accumulate a loss for that example based on the current parameters, compute the derivative of the current cost function \\(w.r.t\\) the activation function, and update our parameters and bias. Finally we take the average of our cost function and our gradients.

Finally, we use our derivatives to update our parameters,

- \\(w_1 := w_1 - \alpha \cdot {dw_1}\\)
- \\(w_2 := w_2 - \alpha \cdot {dw_2}\\)
- \\(b := b - \alpha \cdot {db}\\)

This constitutes __one step__ of gradient descent.

The main problem with this implementation is the nested for loops. For deep learning, which requires very large training sets, _explicit for loops_ will make our implementation very slow. Vectorizing this algorithm will greatly speed up our algorithms running time.

### Vectorization

Vectorization is basically the art of getting ride of explicit for loops. In practice, deep learning requires large datasets (at least to obtain high performance). Explicit for loops lead to computational overhead that significantly slows down the training process.

The main reason vectorization makes such a dramatic difference is that it allows us to take advantage of __parallelization__. The rule of thumb to remember is: _whenever possible, avoid explicit for-loops_.

> In a toy example were \\(n_x\\) is \\(10^6\\), and \\(w, x^{(i)}\\) are random values, vectorization leads to an approximately 300X speed up to compute all \\(z^{(i)}\\)

Lets take a look at some explicit examples:

- Multiple a __matrix__ by a __vector__, e.g., \\(u = Av\\) .

  So, \\(u_i = \sum_jA_{ij}v_j\\) . Instead of using for nested loops, use: `u = np.dot(A,v)`

- Apply exponential operation on every element of a matrix/vector \\(v\\).

  Again, use libraries such as `numpy` to perform this with a single operation, e.g., `u = np.exp(v)`

  > This example applies to almost all operations, `np.log(v)`, `np.abs(v)`, `np.max(v)`, etc...

#### Example: Vectorization of Logistic Regression

##### Forward pass

Lets first review the forward pass of logistic regression for \\(m\\) examples:

\\(z^{(1)} = w^Tx^{(1)} + b\\(;  \\(a^{(1)} = \sigma(z^{1})\\), \\) ...\\) , \\(z^{(m)} = w^Tx^{(m)} + b\\(;  \\(a^{(m)} = \sigma(z^{m})\\)

In logistic regression, we need to compute \\(z^{(i)} = w^Tx^{(i)}+b\\) for each input example \\(x^{(i)}\\) . Instead of using a for loop over each \\(i\\) in range \\((m)\\) we can use a vectorized implementation to compute z directly.

Our vectors are of the dimensions: \\(w \in \mathbb R^{n_x}\\), \\(b \in \mathbb R^{n_x}\\), \\(x \in \mathbb R^{n_x}\\).

Our parameter vector, bias vector, and design matrix are,

\\(w = \begin{bmatrix}w_1 \\\\ ... \\\\ w_{n_x}\end{bmatrix}\\), \\(b = \begin{bmatrix}b_1 \\\\ ... \\\\ b_{n_x}\end{bmatrix}\\), \\(X = \begin{bmatrix}x^{(1)}_1 & ... &x^{(m)} \\\\ ... \\\\ x^{(1)}_{n_x}\end{bmatrix}\\)

So, \\(w^T \cdot X + b = w^Tx^{(i)} + b\\) (for all \\(i\\)). Thus we can compute all \\(w^Tx^{(i)}\\) in one operation if we vectorize!

In numpy code:

`Z = np.dot(w.T,X) + b`

> Note, \\(+ b\\) will perform element-wise addition in python, and is an example of __broadcasting__.

Where \\(Z\\) is a row vector \\([z^{(1)}, ..., z^{(m)}]\\) .

##### Backward pass

Recall, for the gradient computation, we computed the following derivatives:

\\(dz^{(1)} = a^{(1)} - y^{(1)} ... dz^{(m)} = a^{(m)} - y^{(m)}\\)

We define a row vector,

\\(dZ = [dz^{(1)}, ..., dz^{(m)}]\\) .

From which it is trivial to see that,

\\(dZ = A - Y\\), where \\(A = [a^{(1)}, ..., a^{(m)}]\\) and \\(Y = [y^{(1)}, ..., y^{(m)}]\\) .

> This is an element-wise subtraction, \\(a^{(1)} - y^{(1)}, ..., a^{(m)} - y^{(m)}\\) that produces a \\(m\\) length row vector.

We can then compute our _average_ derivatives of the cost function \\(w.r.t\\) to the parameters in two lines of codes,

`db = 1/m * np.sum(dZ)`

`dw = 1/m * np.dot(X, dZ.T)`

Finally, we compare our non-vectorized approach to linear regression vs our vectorized approaches

Non-vectorized Approach       | Vectorized Approach 	|
|:-------------------------:|:-------------------------:|
|![](https://s19.postimg.org/w0z9g6nib/gd_no_vectorization.png)| ![](https://s19.postimg.org/pna6cxawj/gd_vectorization.png.png)|
|Two for loops, one over the training examples \\(x^{(i)}\\) and a second over the features \\(x^{(i)}_j\\) . We have omitted the outermost loop that iterates over gradient steps. | Note that, we still need a single for loop to iterate over each gradient step (regardless if we are using stochastic or mini-batch gradient descent) even in our vectorized approach. |

### Broadcasting

Lets motivate the usefulness of __broadcasting__ with an example. Lets say you wanted to get the percent of total calories from carbs, proteins, and fats for multiple foods.

![food-matrix](https://s19.postimg.org/q01kj1vqb/food_matrix.png.png)

_Can we do this without an explicit for loop?_

Set this matrix to a `(3,4)` numpy matrix `A`.

```
import numy as np

# some numpy array of shape (3,4)
A = np.array([
  [...],
  [...],
  [...]
  ])

cal = A.sum(axis=0) # get column-wise sums
percentage = 100 * A / cal.reshape(1,4) # get percentage of total calories
```

So, we took a `(3,4)` matrix `A` and divided it by a `(1,4)` matrix `cal`. This is an example of broadcasting.

The general principle of broadcast can be summed up as follows:

- \\((m,n) \text{ [+ OR - OR * OR /] } (1, n) \Rightarrow (m,n) \text{ [+ OR - OR * OR /] } (m \text{ copies}, n)\\)
- \\((m,n) \text{ [+ OR - OR * OR /] } (m, 1) \Rightarrow (m,n) \text{ [+ OR - OR * OR /] } (m, n \text{ copies})\\)

Where \\((m, n), (1, n)\\) are matrices, and the operations are performed _element-wise_ after broadcasting.


#### More broadcasting examples

##### Addition

_Example 1_: \\(\begin{bmatrix}1 \\\\ 2 \\\\ 3 \\\\ 4\end{bmatrix} + 100 == \begin{bmatrix}1 \\\\ 2 \\\\ 3 \\\\ 4\end{bmatrix} + \begin{bmatrix}100 \\\\ 100 \\\\ 100 \\\\ 100\end{bmatrix} = \begin{bmatrix}101 \\\\ 102 \\\\ 103 \\\\ 104\end{bmatrix}\\)

_Example 2_: \\(\begin{bmatrix}1 & 2 & 3 \\\\ 4 & 5 & 6\end{bmatrix} + \begin{bmatrix}100 & 200 & 300\end{bmatrix} == \begin{bmatrix}1 & 2 & 3 \\\\ 4 & 5 & 6\end{bmatrix} + \begin{bmatrix}100 & 200 & 300 \\\\ 100 & 200 & 300\end{bmatrix} = \begin{bmatrix}101 & 202 & 303 \\\\ 104 & 205 & 306\end{bmatrix}\\)

_Example 3_: \\(\begin{bmatrix}1 & 2 & 3  \\\\ 4 & 5 & 6\end{bmatrix} + \begin{bmatrix}100 \\\\ 200\end{bmatrix} == \begin{bmatrix}1 & 2 & 3 \\\\ 4 & 5 & 6\end{bmatrix} + \begin{bmatrix}100 & 100 & 100 \\\\ 200 & 200 & 200\end{bmatrix} = \begin{bmatrix}101 & 202 & 303 \\\\ 104 & 205 & 206\end{bmatrix}\\)

### (AISDE) A note on python/numpy vectors

The great flexibility of the python language paired with the numpy library is both a strength and a weakness. It is a strength because of the great expressivity of the pair, but with this comes the opportunity to intro strange, hard-to-catch bugs if you aren't familiar with the intricacies of numpy and in particular broadcasting.

Here are a couple of tips and tricks to minimize the number of these bugs:

- Creating a random array: `a = np.random.randn(5)`
- Arrays of shape `(x, )` are known as __rank 1 array__. They have some nonintuitive properties and don't consistently behave like either a column vector or a row vector. Let `b` be a rank 1 array.
  - `b.T == b`
  - `np.dot(b, b.T)` is a real number, _not the outer product as you might expect_.
- Thus, in this class at least, using rank 1 tensors with an unspecified dimension length is not generally advised. _Always specify both dimensions_.
- If you know the size that your numpy arrays should be in advance, its often useful to throw in a python assertion to help catch strange bugs before they happen:
  - `assert(a.shape == (5,1))`
- Additionally, the reshape function runs in linear time and is thus very cheap to call, use it freely!
   - `a = a.reshape((5,1))`

# Week 3: Shallow neural networks

## Neural network overview

Up until this point, we have used logistic regression as a stand-in for neural networks. The "network" we have been describing looked like:

Network       | Computation Graph 	|
|:-------------------------:|:-------------------------:|
|![](https://s19.postimg.org/3o3rppemr/lr_overview.png)| ![](https://s19.postimg.org/mgfmtblbn/lr_overview_graph.png)|


> \\(a\\) and  \\(\hat y\\) are used interchangeably

A neural network looks something like this:

Network       | Computation Graph 	|
|:-------------------------:|:-------------------------:|
|![](https://s19.postimg.org/77ppfl9nn/nn_overview.png)| ![](https://s19.postimg.org/mt70ziygj/nn_overview_graph.png)|

> We typically don't distinguish between \\(z\\) and \\(a\\) when talking about neural networks, one neuron = one activation = one \\(a\\) like calculation.

We will introduce the notation of superscripting values with \\(^{[l]}\\), where \\(l\\) refers to the layer of the neural network that we are talking about.

> Not to be confused with \\(^{(i)}\\) which we use to refer to a single input example \\(i\\) .

_The key intuition is that neural networks stack activations of inputs multiplied by their weights_.

Similar to the 'backwards' step that we discussed for logistic regression, we will explore the backwards steps that makes learning in a neural network possible.

### Neural network Representation

This is the canonical representation of a neural network

![](https://s19.postimg.org/vbgh3vcoz/neural_network_basics.png)

On the left, we have the __input features__ stacked vertically. This constitutes our __input layer__. The final layer, is called the __output layer__ and it is responsible for generating the predicted value \\(\hat y\\) . Any layer in between this two layers is known as a __hidden layer__. This name derives from the fact that the _true values_ of these hidden units is not observed in the training set.

> The hidden layers and output layers have parameters associated with them. These parameters are denoted \\(W^{[l]}\\) and \\(b^{[l]}\\) for layer \\(l\\) .

Previously, we were referring to our input examples as \\(x^{(i)}\\) and organizing them in a design matrix \\(X\\) . With neural networks, we will introduce the convention of denoting output values of a layer \\(l\\), as a column vector \\(a^{[l]}\\), where \\(a\\) stands for _activation_. You can also think of these as the values a layer \\(l\\) passes on to the next layer.

Another note: the network shown above is a _2-layer_ neural network. We typically do not count the input layer. In light of this, we usually denote the input layer as \\(l=0\\).

### Computing a Neural Networks Output

We will use the example of a single hidden layer neural network to demonstrate the forward propagation of inputs through the network leading to the networks output.

We can think of each unit in the neural network as performing two steps, the _multiplication of inputs by weights and the addition of a bias_, and the _activation of the resulting value_

![](https://s19.postimg.org/qquaof5oz/unit_breakdown.png)

> Recall, that we will use a superscript, \\(^{[l]}\\) to denote values belonging to the \\(l-th\\) layer.

So, the \\(j^{th}\\) node of the \\(l^{th}\\) layer performs the computation

\\[a_j^[{l}] = \sigma(w_i^{[l]^T}a^{[l-1]} + b_i^{[l]})\\]

> Where \\(a^{[l-1]}\\) is the activation values from the precious layer.

for some input \\(x\\). With this notation, we can draw our neural network as follows:

![](https://s19.postimg.org/6i6x39r4j/new_notation_nn.png)

In order to easily vectorize the computations we need to perform, we designate a matrix \\(W^{[l]}\\) for each layer \\(l\\), which has dimensions _(number of units in current layer X number of units in previous layer)_

We can vectorize the computation of \\(z^{[l]}\\) as follows:

![](https://s19.postimg.org/n78cynd9v/vectorized_z_nn.png)

And the computation of \\(a^{[l]}\\) just becomes the element-wise application of the sigmoid function:

![](https://s19.postimg.org/7yifkuh0j/vectorized_a_nn.png)

We can put it all together for our two layer neural network, and outline all the computations using our new notation:

![](https://s19.postimg.org/5so4qvgab/putting_it_all_together_new_notation.png)

### Vectorizing across multiple examples

In the last video, we saw how to compute the prediction for a neural network with a single input example. In this video, we introduce a vectorized approach to compute predictions for many input examples.  

We have seen how to take a single input example \\(x\\) and compute \\(a^{[2]} = \hat y\\) for a 2-layered neural network. If we have \\(m\\) training examples, we can used a vectorized approach to compute all \\(m\\) predictions.

First, lets introduce a new notation. The activation values of layer \\(l\\) for input example \\(i\\) is:

\\[a^{[l](i)}\\]

The \\(m\\) predictions our 2-layered are therefore computed in the following way:

![](https://s19.postimg.org/mt70zhvvn/m_examples_nn.png)

Recall that \\(X\\) is a \\((n_x, m)\\) design matrix, where each column is a single input example and \\(W^{[l]}\\) is a matrix where each row is the transpose of the parameter column vector for layer \\(l\\).

Thus, we can now compute the activation of a layer in the neural network for all training examples:

\\[Z^{[l]} = W^{[l]}X + b^{[l]}\\]
\\[A^{[l]} = sign(Z^{[l]})\\]

As an example, the result of a matrix multiplication of \\(W^{[1]}\\) by \\(X\\) is a matrix with dimensions \\((j, m)\\) where \\(j\\) is the number of units in layer \\(1\\) and \\(m\\) is the number of input examples

![](https://s19.postimg.org/6w892blcj/WX_vector.jpg)

\\(A^{[l]}\\) is therefore a matrix of dimensions (size of layer \\(l\\) X \\(m\\)). The top-leftmost value is the activation for the first unit in the layer \\(l\\) for the first input example \\(i\\), and the bottom-rightmost value is the activation for the last unit in the layer \\(l\\) for the last input example \\(m\\) .

![](https://s19.postimg.org/dmoqbqt2r/vectorized_activations.png)

## Activation Functions

So far, we have been using the __sigmoid__ activation function

\\[\sigma(z) = \frac{1}{1 + e^{-z}}\\]

It turns out there are much better options.

### Tanh

The __hyperbolic tangent function__ is a non-linear activation function that almost always works better than the sigmoid function.

\\[tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}\\]

![](https://upload.wikimedia.org/wikipedia/commons/thumb/7/76/Sinh_cosh_tanh.svg/640px-Sinh_cosh_tanh.svg.png?1514655794955.png)

> The tanh function is really just a shift of the sigmoid function so that it crosses through the origin.

The tanh activation usually works better than sigmoid activation function for hidden units because the mean of its output is closer to zero, and so it centers the data better for the next layer.

The single exception of sigmoid outperforming tanh is when its used in the ouput layer. In this case, it can be more desirable to scale our outputs from \\(0\\) to \\(1\\) (particularly in classification, when we want to output the probability that something belongs to a certain class). Indeed, we often mix activation functions in neural networks, and denote them:

\\[g^{[p]}(z)\\]

Where \\(p\\) is the \\(p^th\\) activation function.

If \\(z\\) is either very large, or very small, the derivative of both the tanh and sigmoid functions becomes very small, and this can slow down learning.

### ReLu

The __rectified linear unit__ activation function solves the disappearing gradient problem faced by tanh and sigmoid activation functions. In practice, it also leads to faster learning.

\\[ReLu(z) = max(0, z)\\]

![](https://upload.wikimedia.org/wikipedia/en/thumb/6/6c/Rectifier_and_softplus_functions.svg/640px-Rectifier_and_softplus_functions.svg.png?1514655837364.png)

> Note: the derivative at exactly 0 is not well-defined. In practice, we can simply set it to 0 or 1 (it matters little, due to the unlikeliness of a floating point number to ever be \\(0.0000...\\) exactly).

One disadvantage of ReLu is that the derivative is equal to \\(0\\) when \\(z\\) is negative. __Leaky ReLu__'s aim to solve this problem with a slight negative slope for values of \\(z<0\\) .

\\[ReLu(z) = max(0.01 * z, z)\\]

![](http://lamda.nju.edu.cn/weixs/project/CNNTricks/imgs/leaky.png)

> Image sourced from [here](http://lamda.nju.edu.cn/weixs/project/CNNTricks/imgs/leaky.png).

Sometimes, the \\(0.01\\) value is treated as an adaptive parameter of the learning algorithm. Leaky ReLu's solve a more general problem of "[dead neurons](https://www.quora.com/What-is-the-definition-of-a-dead-neuron-in-Artificial-Neural-Networks?share=1)". However, it is not used as much in practice.

#### Rules of thumb for choosing activations functions

- _If your output is a 0/1 value_, i.e., you are performing binary classification, the sigmoid activation is a natural choice for the output layer.
- _For all other units_, ReLu's is increasingly the default choice of activation function.

#### Why do you need non-linear activation functions?

We could imagine using some __linear__ activation function, \\(g(z) = z\\) in place of the __non-linear__ activation functions we have been using so far. Why is this a bad idea? Lets illustrate out explanation using our simple neural networks

![](https://s19.postimg.org/vbgh3vcoz/neural_network_basics.png)

For this linear activation function, the activations of our simple network become:

\\[z^{[1]} = W^{[1]}x + b^{[1]}\\]
\\[a^{[1]} = z^{[1]}\\]
\\[z^{[2]} = W^{[2]}x + b^{[2]}\\]
\\[a^{[2]} = z^{[2]}\\]

From which we can show that,

\\[a^{[2]} = (W^{[2]}W^{[1]})x + (W^{[2]}b^{[1]} + b^{[2]})\\]
\\[a^{[2]} = W'x + b' \text{, where } W' = W^{[2]}W^{[1]} \text{ and } b' = W^{[2]}b^{[1]} + b^{[2]}\\]

Therefore, in the case of a _linear activation function_, the neural network is outputting a _linear function of the inputs_, no matter how many hidden layers!

##### Exceptions

There are (maybe) two cases in which you may actually want to use a linear activation function.

1. The output layer of a network used to perform regression, where we want \\(\hat y\\) to be a real-valued number, \\(\hat y \in \mathbb R\\)
2. Extremely specific cases pertaining to compression.

### Derivatives of activation functions

When perform back-propogation on a network, we need to compute the derivatives of the activation functions. Lets take a look at our activation functions and their derivatives

#### Sigmoid

![](https://s19.postimg.org/dy66p1jyr/sigmoid_deriv.png)

The deriviative of \\(g(z)\\), \\(g(z)'\\) is:

\\[\frac{d}{dz}g(z) = \frac{1}{1 + e^{-z}}(1 - \frac{1}{1 + e^{-z}})= g(z)(1-g(z)) = a(1-a)\\]

> We can sanity check this by inputting very large, or very small values of \\(z\\) into our derivative formula and inspecting the size of the outputs.

Notice that if we have already computed the value of \\(a\\), we can very cheaply compute the value of \\(g(z)'\\) .

#### Tanh

![](https://s19.postimg.org/g2qjq510z/tanh_deriv.png)

The deriviative of \\(g(z)\\), \\(g(z)'\\) is:

\\[\frac{d}{dz}g(z) = 1 - (tanh(z))^z\\]

> Again, we can sanity check this inspecting that the outputs for different values of \\(z\\) match our intuition about the activation function.

#### ReLu

![](https://s19.postimg.org/i7awr6scz/relu_deriv.png)

The deriviative of \\(g(z)\\), \\(g(z)'\\) is:

\\[\frac{d}{dz}g(z) = 0 \text{ if } z < 0 ; 1 \text{ if } z > 0; \text{ undefined if } z = 0\\]

> If \\(z = 0\\), we typically default to setting \\(g(z)\\) to either \\(0\\) or \\(1\\) . In practice this matters little.

## Gradient descent for Neural Networks

Lets implement gradient descent for our simple 2-layer neural network.

Recall, our parameters are: \\(W^{[1]}, b^{[1]}, W^{[2]}, b^{[2]}\\) . We have number of features, \\(n_x = n^{[0]}\\), number of hidden units \\(n^{[1]}\\), and \\(n^{[2]}\\) output units.

Thus our dimensions:

- \\(W^{[1]}\\) : (\\(n^{[1]}, n^{[0]}\\))
- \\(b^{[1]}\\) : (\\(n^{[1]}, 1\\))
- \\(W^{[2]}\\) : (\\(n^{[2]}, n^{[1]}\\))
- \\(b^{[2]}\\) : (\\(n^{[2]}, 1\\))

Our cost function is: \\(J(W^{[1]}, b^{[1]}, W^{[2]}, b^{[2]}) = \frac{1}{m}\sum_{i=1}^m \ell(\hat y, y)\\)

> We are assuming binary classification.

__Gradient Descent sketch__

1. Initialize parameters _randomly_
2. Repeat:
    - compute predictions \\(\hat y^{(i)}\\) for \\(i = 1 ,..., m\\)
    - \\(dW^{[1]} = \frac{\partial J}{\partial W^{[1]}}, db^{[1]} = \frac{\partial J}{\partial b^{[1]}}, ...\\)
    - \\(W^{[1]} = W^{[1]} - \alpha dW^{[1]}, ...\\)
    - \\(b^{[1]} = b^{[1]} - \alpha db^{[1]}, ...\\)

The key to gradient descent is to computation of the derivatives, \\(\frac{\partial J}{\partial W^{[l]}}\\) and \\(\frac{\partial J}{\partial b^{[l]}}\\) for all layers \\(l\\) .

### Formulas for computing derivatives

We are going to simply present the formulas you need, and defer their explanation to the next video. Recall the computation graph for our 2-layered neural network:

![](https://s19.postimg.org/mt70ziygj/nn_overview_graph.png)

And the vectorized implementation of our computations in our __forward propagation__

1.\\[Z^{[1]} = W^{[1]}X + b^{[1]}\\]
2.\\[A^{[1]} = g^{[1]}(z^{[1]})\\]
3.\\[Z^{[2]} = W^{[2]}X + b^{[2]}\\]
4.\\[A^{[2]} = g^{[2]}(z^{[2]})\\]

> Where \\(g^{[2]}\\) would likely be the sigmoid function if we are doing binary classification.

Now we list the computations for our __backward propagation__

1.\\[dZ^{[2]} = A^{[2]} - Y\\]
2.\\[dW^{[2]} = \frac{1}{m}dZ^{[2]}A^{[1]T}\\]

> Transpose of A accounts for the fact that W is composed of transposed column vectors of parameters.

3.\\[db^{[2]} = \frac{1}{m}np.sum(dZ^{[2]}, axis = 1, keepdims=True)\\]

> Where \\(Y = [y^{(1)}, ..., y^{[m]}]\\) . The `keepdims` arguments prevents numpy from returning a rank 1 array, \\((n,)\\)

4.\\[dZ^{[1]} = W^{[2]T}dZ^{[2]} \odot g(Z)' (Z^{[1]})\\]

> Where \\(\odot\\) is the element-wise product. Note: this is a collapse of \\(dZ\\) and \\(dA\\) computations.

5.\\[dW{[1]} = \frac{1}{m} = dZ^{[1]}X^T\\]
6.\\[db^{[1]} = \frac{1}{m}np.sum(dZ^{[1]}, axis=1, keepdims=True)\\]

## Random Initialization

When you train your neural network, it is important to initialize your parameters _randomly_. With logistic regression, we were able to initialize our weights to _zero_ because the cost function was convex. We will see that this _will not work_ with neural networks.

Lets take the following network as example:

![](https://s19.postimg.org/8b9tr0jur/super_simple_network.png)

Lets say we initialize our parameters as follows:

\\(W^{[1]} = \begin{bmatrix}0 & 0 \\\\ 0 & 0 \end{bmatrix}\\), \\(b^{[1]} = \begin{bmatrix} 0 \\\\ 0 \end{bmatrix}\\),
\\(W^{[2]} = \begin{bmatrix} 0 & 0 \\\\ 0 & 0 \end{bmatrix}\\),
\\(b^{[2]} = \begin{bmatrix} 0 \\\\ 0 \end{bmatrix}\\)

> It turns out that initializing the bias \\(b\\) with zeros is OK.

The problem with this initialization is that for any input examples \\(i, j\\),

\\[a^{[1]}_i == a^{[1]}_j\\]

Similarly,

\\[dz^{[1]}_i == dz^{[1]}_j\\]

Thus, \\(dW^{[1]}\\) will be some matrix \\(\begin{bmatrix}u & v \\\\ u & v\end{bmatrix}\\) and all updates to the parameters \\(W^{[1]}\\) will be identical.

> Note we are referring to our single hidden layer \\(^{[1]}\\) but this would apply to any hidden layer of any fully-connected network, no matter how large.

Using a _proof by induction_, it is actually possible to prove that after any number of rounds of training the two hidden units are still computing _identical functions_. This is often called the __symmetry breaking problem__.

The solution to this problem, is to initialize parameters _randomly_. Heres an example on how to do that with numpy:

- \\(W^{[1]}\\) = `np.random.rand(2,2) * 0.01`
- \\(W^{[2]}\\) = `np.random.rand(1,2) * 0.01`
- ...

> This will generate small, gaussian random values.

- \\(b^{[1]}\\) = `np.zeros((2,1))`
- \\(b^{[2]}\\) = `0`
- ...

> In next weeks material, we will talk about how and when you might choose a different factor than \\(0.01\\) for initialization.

It turns out the \\(b\\) does not have this symmetry breaking problem, because as long as the hidden units are computing different functions, the network will converge on different values of \\(b\\), and so it is fine to initialize it to zeros.

#### Why do we initialize to small values?

For a _sigmoid-like_ activation function, large parameter weights (positive or negative) will make it more likely that \\(z\\) is very large (positive or negative) and thus \\(dz\\) will approach \\(0\\), _slowing down learning dramatically_.

> Note this is a less of an issue when using ReLu's, however many classification problems use sigmoid activations in their output layer.

# Week 4: Deep Neural Networks

## What is a deep neural network?

A deep neural network is simply a network with more than 1 hidden layer. Compared to logistic regression or a simple neural network with one hidden layer (which are considered __shallow__ models) we say that a neural network with many hidden layers is a __deep__ model, hence the terms _deep learning / deep neural networks_.

> Shallow Vs. deep is a matter of degree, the more hidden layers, the deeper the model.

![]()

Over the years, the machine learning and AI community has realized that deep networks are excellent function approximators, and are able to learn incredibly complex functions to map inputs to outputs.

> Note that is difficult to know in advance how deep a neural network needs to be to learn an effective mapping.

### Notation

Lets go over the notation we will need using an example network

![]()

- We will use \\(L\\) to denote the number of layers in the network (in this network \\(L=4\\))
- \\(n^{[l]}\\) denotes the number of units in layers \\(l\\) (for example, in this network \\(n^{[1]} = 5\\))
- \\(a^{[l]}\\) denotes the activations in layer \\(l\\)
- \\(W^{[l]}\\), \\(b^{[l]}\\) denotes the weights and biases for \\(z^{[l]}\\)
- \\(x = a^{[0]}\\) and \\(\hat y = a^{[L]}\\)
