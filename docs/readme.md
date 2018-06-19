# Course 1: Neural Networks and Deep Learning

#### TOC

1. [Week 1: Introduction](#week-1-introduction)
2. [Week 2: Neural networks basics](#week-2-neural-networks-basics)
3. [Week 3: Shallow neural networks](#week-3-shallow-neural-networks)
4. [Week 4: Deep Neural Networks](#week-4-deep-neural-networks)

#### Resources

- [Notation cheetsheet](https://d3c33hcgiwev3.cloudfront.net/_106ac679d8102f2bee614cc67e9e5212_deep-learning-notation.pdf?Expires=1514764800&Signature=bnnZZMJUAG2PZPWezLLN6EeKjihlTdaVPo1fqHDdsPmXkLDyjVG-fBtstgOCIcFkKd8OGx845pIKDITTFGm0sMA1eGo4lAIqP7Btffy5VGBRwasKW3WCGGkP-dmq0Vw7Y83ezax4wQCzzYB6iPevY8QniePzg-iq~O5a9hJ4TRk_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A) (I recommend printing this out and sticking it on the wall where you work!)
- Check links at end of all programming assignments, these are good resources.

## Week 1: Introduction

### What is a neural network?

#### Supervised Learning with Neural Networks

In supervised learning, you have some input \\(x\\) and some output \\(y\\) . The goal is to learn a mapping \\(x \rightarrow y\\) .

Possibly, the single most lucrative (but not the most inspiring) application of deep learning today is online advertising. Using information about the ad combined with information about the user as input, neural networks have gotten very good at predicting whether or not you click on an ad. Because the ability to show you ads that you're more likely to click on has a _direct impact on the bottom line of some of the very large online advertising companies_.

Here are some more areas in which deep learning has had a huge impact:

- __Computer vision__ the recognition and classification of objects in photos and videos.
- __Speech Recognition__ converting speech in audio files into transcribed text.
- __Machine translation__ translating one natural language to another.
- __Autonomous driving__

[![supervised_learning.png](https://s19.postimg.cc/wdqnmh0o3/supervised_learning.png)](https://postimg.cc/image/r2br1relb/)

A lot of the value generation from using neural networks have come from intelligently choosing our \\(x\\) and \\(y\\) and learning a mapping.

We tend to use different architectures for different types of data. For example, __convolutional neural networks__ (CNNs) are very common for _image data_, while __recurrent neural networks__ (RNNs) are very common for _sequence data_ (such as text). Some data, such as radar data from autonomous vehicles, don't neatly fit into any particularly category and so we typical use a complex/hybrid network architecture.

#### Structured vs. Unstructured Data

You can think of __structured data__ as essentially meaning _databases of data_. It is data that is highly _structured_, typically with multiple, well-defined attributes for each piece of data. For example, in housing price prediction, you might have a database where the columns tells you the size and the number of bedrooms. Or in predicting whether or not a user will click on an ad, you might have information about the user, such as the age, some information about the ad, and then labels why that you're trying to predict.

In contrast, __unstructured data__ refers to things like audio, raw audio,
or images where you might want to recognize what's in the image or text. Here the features might be the pixel values in an image or the individual words in a piece of text. Historically, it has been much harder for computers to make sense of unstructured data compared to structured data. In contrast the human race has evolved to be very good at understanding audio cues as well as images. _People are really good at interpreting unstructured data_. And so one of the most exciting things about the rise of neural networks is that, thanks to deep learning, thanks to neural networks, computers are now much better at interpreting unstructured data as well compared to just a few years ago. And this creates opportunities for many new exciting applications that use speech recognition, image recognition, natural language processing on text.

Because people have a natural empathy to understanding unstructured data, you might hear about neural network successes on unstructured data more in the media because it's just cool when the neural network recognizes a cat. We all like that, and we all know what that means. But it turns out that a lot of short term economic value that neural networks are creating has also been on structured data, such as much better advertising systems, much better profit recommendations, and just a much better ability to process the giant databases that many companies have to make accurate predictions from them.

 [![unstructured_vs_structured_data.png](https://s19.postimg.cc/66pgpyd37/unstructured_vs_structured_data.png)](https://postimg.cc/image/5ty2jrutb/)

### Why is Deep Learning taking off?

_If the basic technical details surrounding deep learning have been around for decades, why are they just taking off now?_

First and foremost, the massive amount of (labeled) data we have been generating for the past couple of decades (in part because of the 'digitization' of our society).

It turns out, that large, complex neural networks can take advantage of these huge data stores. Thus, we often say _scale_ has been driving progress with deep learning, where scale means the size of the data, the size/complexity of the neural network, and the growth in computation.

The interplay between these 'scales' is apparent when you consider that many of the algorithmic advances of neural networks have come from making them more computational efficient.  

__Algorithmic Advances: ReLu__

One of the huge breakthroughs in neural networks has been the seemingly simple switch from the __sigmoid__ activation function to the __rectified linear__ (ReLu) activation function.

One of the problems with using __sigmoid__ functions is that its gradients approach 0 as input to the sigmoid function approaches and \\(+\infty\\) and \\(-\infty\\) . In this case, the updates to the parameters become very small and our learning slows dramatically.

With ReLu units, our gradient is equal to \\(1\\) for all positive inputs. This makes learning with gradient descent much faster. See [here](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)) for more information on ReLu's.

![](https://upload.wikimedia.org/wikipedia/commons/thumb/6/6c/Rectifier_and_softplus_functions.svg/640px-Rectifier_and_softplus_functions.svg.png?1527957850905)

__Scale Advances__

With smaller training sets, the relative ordering of the algorithms is actually not very well defined so if you don't have a lot of training data it is often up to your skill at hand engineering features that determines the
performance. For small training sets, it's quite possible that if someone training an SVM is more motivated to hand engineer features they will outperform a powerful neural network architecture.

 [![scale.png](https://s19.postimg.cc/6i6x39jer/scale.png)](https://postimg.cc/image/t6w42u0sf/)

However, for very large training sets, _we consistently see large neural networks dominating the other approaches_.

## Week 2: Neural networks basics

### Binary Classification

First, some notation,

- \\(n\\) is the number of data attributes, or _features_
- \\(m\\) is the number of input examples in our dataset (sometimes we write \\(m_{train}, m_{test}\\) to be more explicit).
- our data is represented as input, output pairs, \\((x^{(1)},y^{(1)}), ..., (x^{(m)},y^{(m)})\\) where \\(x \in \mathbb R^n\\) , \\(y \in \{0,1\}\\)
- \\(X\\) is our design matrix, which is simply columns of our input vectors \\(x^{(i)}\\) , thus it has dimensions of \\(n\\) x \\(m\\) .
- \\(Y = [y^{(1)}, ..., y^{(m)}]\\) , and is thus a \\(1\\) x \\(m\\) matrix.

> Note, this is different from many other courses which represent the design matrix, \\(X\\) as rows of transposed input vectors and the output vector \\(Y\\) as a \\(m\\) x \\(1\\) column vector. The above convention turns out to be easier to implement.

When programming neural networks, implementation details become extremely important (_e.g_. vectorization in place of for loops).

We are going to introduce many of the key concepts of neural networks using __logistic regression__, as this will make them easier to understand. Logistic regression is an algorithm for __binary classification__. In binary classification, we have an input (_e.g_. an image) that we want to classifying as belonging to one of two classes.

#### Logistic Regression (Crash course)

Given an input feature vector \\(x\\) (perhaps corresponding to an images flattened pixel data), we want \\(\hat y\\) , the probability of the input examples class, \\(\hat y = P(y=1 | x)\\)

> If \\(x\\) is a picture, we want the chance that this is a picture of a cat, \\(\hat y\\) .

The parameters of our model are \\(w \in \mathbb R^{n_x}\\) , \\(b \in \mathbb R\\) . Our output is \\(\hat y = \sigma(w^Tx + b)\\) were \\(\sigma\\) is the __sigmoid function__.

![sigmoid](https://upload.wikimedia.org/wikipedia/commons/thumb/8/88/Logistic-curve.svg/640px-Logistic-curve.svg.png)

The formula for the sigmoid function is given by: \\(\sigma(z) = \frac{1}{1 + e^{-z}}\\) where \\(z = w^Tx + b\\) . We notice a few things:

- If \\(z\\) is very large, \\(e^{-z}\\) will be close to \\(0\\) , and so \\(\sigma(z)\\) is very close to \\(1\\) .
- If \\(z\\) is very small, \\(e^{-z}\\) will grow very large, and so \\(\sigma(z)\\) is very close to \\(0\\) .

> It helps to look at the plot \\(y = e^{-x}\\)

Thus, logistic regression attempts to learn parameters which will classify images based on their probability of belonging to one class or the other. The classification decision is decided by applying the sigmoid function to \\(w^Tx + b\\) .

> Note, with neural networks, it is easier to keep the weights \\(w\\) and the biases \\(b\\) separate. Another notation involves adding an extra parameters (\\(w_0\\) which plays the role of the bias.

__Loss function__

Our prediction for a given example \\(x^{(i)}\\) is \\(\hat y^{(i)} = \sigma(w^Tx^{(i)} + b)\\) .

We chose __loss function__, \\(\ell(\hat y, y) = -(y \; log\; \hat y + (1-y) \;log(1-\hat y))\\).

We note that:

- If \\(y=1\\) , then the loss function is \\( \ell(\hat y, y) = -log\; \hat y\\) . Thus, the loss approaches zero as \\(\hat y\\) approaches 1.
- If \\(y=0\\) , then the loss function is \\(\ell(\hat y, y) = -log\; (1 -\hat y)\\) . Thus, the loss approaches zero as \\(\hat y\\) approaches 0.

> Note, while \\(\ell_2\\) loss is taught in many courses and seems like an appropriate choice, it is non-convex and so we cannot use gradient descent to optimize it.

> An optional video is given further justifying the use of this loss function. Watch it and add notes here!

Note that the __loss function__ measures how well we are doing on a _single example_. We now define a __cost function__, which captures how well we are doing on the entire dataset:

\\(J(w,b) = \frac{1}{m}\sum^m_{i=1} \ell(\hat y^{(i)}, y^{(i)}) = - \frac{1}{m}\sum^m_{i=1}(y^{(i)} \; log\; \hat y^{(i)} + (1-y^{(i)}) \;log(1-\hat y^{(i)}))\\)

> Note that this notation is somewhat unique, typically the cost/loss functions are just interchangeable terms. However in this course, we will define the __loss function__ as computing the error for a single training example and the __cost function__ as the average of the loss functions of the entire training set.

#### Gradient Descent

We want to find \\(w,b\\) which minimize \\(J(w,b)\\) . We can plot the __cost function__ with \\(w\\) and \\(b\\) as our horizontal axes:

[![cost_surface.png](https://s19.postimg.cc/pna6cuy0z/cost_surface.png)](https://postimg.cc/image/a1suswm2n/)

> In practice, \\(w\\) typically has many more dimensions.

Thus, the cost function \\(J(w,b)\\) can be thought of as a surface, were the height of the surface above the horizontal axes is its value. We want to find the values of our parameters \\(w, b\\) at the lowest point of this surface, the point at which the average loss is at its minimum.

__Gradient Descent Algorithm__

Initialize \\(w,b\\) to some random values

> because this cost function is convex, it doesn't matter what values we use to initialize, \\(0\\) is usually chosen for logistic regression.

Repeat

1. \\(w := w - \alpha \frac{dJ(w)}{dw}\\)
2. \\(b := b - \alpha \frac{dJ(w)}{db}\\)

> \\(\alpha\\) is our learning rate, it controls how big a step we take on each iteration. Some notation, typically we use \\(\partial\\) to denote the partial derivative of a function with \\(2\\) or more variables, and \\(d\\) to denote the derivative of a function of only \\(1\\) variable.

[![gradient_descent.png](https://s19.postimg.cc/a1susyr8j/gradient_descent.png)](https://postimg.cc/image/y5jmh99pb/)

When implementing gradient descent in code, we will use the variable \\(dw\\) to represent \\(\frac{dJ(w, b)}{dw}\\) (this size of the step for \\(w\\) and \\(db\\) to represent \\(\frac{dJ(w, b)}{db}\\) (the size of the step for \\(b\\) .

#### (ASIDE) Calculus Review

__Intuition about derivatives__

###### Linear Function Example

Take the function \\(f(a) = 3a\\). Then \\(f(a) = 6\\) when \\(a = 2\\) . If we were to give \\(a\\) a tiny nudge, say to \\(a = 2.001\\) , what happens to \\(f(a)\\) ?

[![derivative.png](https://s19.postimg.cc/wdqnmadgz/derivative.png)](https://postimg.cc/image/4qdy86sa7/)

Then \\(f(a) = 6.003\\) , but more importantly if we inspect the triangle formed by performing the nudge, we can get the slope of the function between \\(a\\) and \\(a + 0.001\\) as the \\(\frac{height}{width} = 3\\) .

Thus, the __derivative__ (or slope) of \\(f(a)\\) _w.r.t_ \\(a\\) is \\(3\\) . We say that \\(\frac{df(a)}{da} = 3\\) or \\(\frac{d}{da}f(a) = 3\\)

> Add my calculus notes here!
> Link to BlueBrown videos.

###### Non-Linear Function Example

Take the function \\(f(a) = a^2\\) . Then \\(f(a) = 4\\) when \\(a = 2\\) . If we were to give \\(a\\) a tiny nudge, say to \\(a = 2.001\\), what happens to \\(f(a)\\)?

[![more_derivatives.png](https://s19.postimg.cc/535ceh5g3/more_derivatives.png)](https://postimg.cc/image/89zvy3pvz/)

Then \\(f(a) = 4.004\\), but more importantly if we inspect the triangle formed by performing the nudge, we can get the slope of the function between \\(a\\) and \\(a + 0.001\\) as the \\(\frac{height}{width} = 4\\) .

In a similar way, we can perform this analysis for any point \\(a\\) on the plot, and we will see that slope of \\(f(a)\\) at some point \\(a\\) is equal to \\(2a\\) .

Thus, the __derivative__ (or slope) of \\(f(a)\\) _w.r.t_ \\(a\\) is \\(2a\\) . We say that \\(\frac{df(a)}{da} = 2a\\) or \\(\frac{d}{da}f(a) = 2a\\) .

#### Computation Graph

A __computation graph__ organizes a series of computations into left-to-right and right-to-left passes. Lets build the intuition behind a computation graph.

Say we are trying to compute a function \\(J(a,b,c) = 3(a + bc)\\) . This computation of this function actually has three discrete steps:

- compute \\(u = bc\\)
- compute \\(v = a + u\\)
- compute J = \\(3v\\)

We can draw this computation in a graph:

[![computation_graph.png](https://s19.postimg.cc/qcsyp86ab/computation_graph.png)](https://postimg.cc/image/r2br1l6tr/)

The computation graph is useful when you have some variable or output variable that you want to optimize (\\(J\\) in this case, in logistic regression it would be our _cost function output_). A _forward pass_ through the graph is represented by _left-to-right_ arrows (as drawn above) and a _backwards pass_ is represented by _right-to-left_ arrows.

A backwards pass is a natural way to represent the computation of our derivatives.  

__Derivatives with a computation graph__

Lets take a look at our computation graph, and see how we can use it to compute the partial derivatives of \\(J\\) i.e., lets carry out backpropogation on this computation graph by hand.

> Informally, you can think of this as asking: "If we were to change the value of \\(v\\) slightly, how would \\(J\\) change?"

[![clean_computation_graph.png](https://s19.postimg.cc/q01kj10v7/clean_computation_graph.png)](https://postimg.cc/image/iwtp3evfj/)

First, we use our informal way of computing derivatives, and note that a small change to \\(v\\) results in a change to \\(J\\) of 3X that small change, and so \\(\frac{dJ}{dv} = 3\\) . This represents one step in our backward pass, the first step in backpropagation.

Now let's look at another example. What is \\(\frac{dJ}{da}\\)?

We compute the \\(\frac{dJ}{da}\\) from the second node in the computation graph by noting that a small change to a results ina change to \\(J\\) of 3X that small  change, and so \\(\frac{dJ}{da} = 3.\\) This represents our second step in our backpropagation.

One way to break this down is to say that by changing \\(a\\), we change \\(v\\), the magnitude of this change is \\(\frac{dv}{da}\\) . Through this change in \\(v\\), we also change \\(J\\), and the magnitude of the change is \\(\frac{dJ}{dv}\\) . To captue this more generally, we use the __chain rule__ from calculus, informally:

\\[\text{if } a \rightarrow v \rightarrow J \text{, then } \frac{dJ}{da} = \frac{dJ}{dv} \frac{dv}{da}\\]

> Here, just take \\(\rightarrow\\) to mean 'effects'. A formal definition of the chain rule can be found [here](https://en.wikipedia.org/wiki/Chain_rule).

The amount \\(J\\) changes when you when you nudge \\(a\\) is the product of the amount \\(J\\) changes when you nudge \\(v\\) multiplied by the amount \\(v\\) changes when you nudge \\(a\\) .

> __Implementation note__: When writing code to implement backpropagation, there is typical a single output variable you want to optimize, \\(dvar\\), (the value of the cost function). We will follow to notation of calling this variable \\(dvar\\) .

If we continue performing backpropagation steps, we can determine the individual contribution a change to the input variables has on the output variable. For example,

\\[\frac{dJ}{db} = \frac{dJ}{du} \frac{du}{db} = (3)(2) = 6\\]

The key take away from this video is that when computing derivatives to determine the contribution of input variables to change in an output variable, the most efficient way to do so is through a right to left pass through a computation graph. In particular, we'll first compute the derivative with respect to the output of the left-most node in a backward pass, which becomes useful for computing the derivative with respect to the next node and so forth. The __chain rule__ makes the computation of these derivatives tractable.

#### Logistic Regression Gradient Descent

Logistic regression recap:

\\[z = w^Tx + b\\]
\\[\hat y = a = \sigma(z)\\]
\\[\ell(a,y) = -(ylog(a) + (1-y)log(1-a))\\]

> \\(\ell\\) is our loss for a single example, and \\(\hat y\\) are our predictions.

For this example, lets assume we have only two features: \\(x_1\\), \\(x_2\\) . Our computation graph is thus:

[![computation_graph_logression.png](https://s19.postimg.cc/jz3vlzgtv/computation_graph_logression.png)](https://postimg.cc/image/le5gaphwv/)


Our goal is to modify the parameters to minimize the loss \\(\ell\\) . This translates to computing derivatives \\(w.r.t\\) the loss function. Following our generic example above, we can compute all the relevant derivatives using the chain rule. The first two passes are computed by the following derivatives:

1. \\(\frac{d\ell(a,y)}{da} = - \frac{y}{a} + \frac{1-y}{1-a}\\)
2. \\(\frac{d\ell(a,y)}{dz} = \frac{d\ell(a,y)}{da} \cdot \frac{da}{dz} =  a - y\\)

> Note: You should prove these to yourself.

> __Implementation note__, we use \\(dx\\) as a shorthand for \\(\frac{d\ell(\hat y,y)}{dx}\\) for some variable \\(x\\) when implementing this in code.


Recall that the final step is to determine the derivatives of the loss function \\(w.r.t\\) to the parameters.

- \\(\frac{d\ell(a,y)}{dw_1} = x_1 \cdot \frac{d\ell(a,y)}{dz}\\)
- \\(\frac{d\ell(a,y)}{dw_2} = x_2 \cdot \frac{d\ell(a,y)}{dz}\\)

One step of gradient descent would perform the updates:

- \\(w_1 := w_1 - \alpha \frac{d\ell(a,y)}{dw_1}\\)
- \\(w_2 := w_2 - \alpha \frac{d\ell(a,y)}{dw_2}\\)
- \\(b := b - \alpha \frac{d\ell(a,y)}{db}\\)

__Extending to \\(m\\) examples__

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

#### Vectorization

Vectorization is basically the art of getting ride of explicit for loops. In practice, deep learning requires large datasets (at least to obtain high performance). Explicit for loops lead to computational overhead that significantly slows down the training process.

The main reason vectorization makes such a dramatic difference is that it allows us to take advantage of __parallelization__. The rule of thumb to remember is: _whenever possible, avoid explicit for-loops_.

> In a toy example were \\(n_x\\) is \\(10^6\\), and \\(w, x^{(i)}\\) are random values, vectorization leads to an approximately 300X speed up to compute all \\(z^{(i)}\\)

Lets take a look at some explicit examples:

- Multiple a __matrix__ by a __vector__, e.g., \\(u = Av\\) .

  So, \\(u_i = \sum_jA_{ij}v_j\\) . Instead of using for nested loops, use: `u = np.dot(A,v)`

- Apply exponential operation on every element of a matrix/vector \\(v\\).

  Again, use libraries such as `numpy` to perform this with a single operation, e.g., `u = np.exp(v)`

  > This example applies to almost all operations, `np.log(v)`, `np.abs(v)`, `np.max(v)`, etc...

__Example: Vectorization of Logistic Regression__

###### Forward pass

Lets first review the forward pass of logistic regression for \\(m\\) examples:

\\(z^{(1)} = w^Tx^{(1)} + b\\);  \\(a^{(1)} = \sigma(z^{1})\\), \\( ...\\) , \\(z^{(m)} = w^Tx^{(m)} + b\\);  \\(a^{(m)} = \sigma(z^{m})\\)

In logistic regression, we need to compute \\(z^{(i)} = w^Tx^{(i)}+b\\) for each input example \\(x^{(i)}\\) . Instead of using a for loop over each \\(i\\) in range \\((m)\\) we can use a vectorized implementation to compute z directly.

Our vectors are of the dimensions: \\(w \in \mathbb R^{n_x}\\), \\(b \in \mathbb R^{n_x}\\), \\(x \in \mathbb R^{n_x}\\).

Our parameter vector, bias vector, and design matrix are,

\\(w = \begin{bmatrix}w_1 \\\\ ... \\\\ w_{n_x}\end{bmatrix}\\), \\(b = \begin{bmatrix}b_1 \\\\ ... \\\\ b_{n_x}\end{bmatrix}\\), \\(X = \begin{bmatrix}x^{(1)}_1 & ... & x^{(m)} \\\\ ... \\\\ x^{(1)}_{n_x}\end{bmatrix} \\)

So, \\(w^T \cdot X + b = w^Tx^{(i)} + b\\) (for all \\(i\\)). Thus we can compute all \\(w^Tx^{(i)}\\) in one operation if we vectorize!

In numpy code:

`Z = np.dot(w.T,X) + b`

> Note, \\(+ b\\) will perform element-wise addition in python, and is an example of __broadcasting__.

Where \\(Z\\) is a row vector \\([z^{(1)}, ..., z^{(m)}]\\) .

###### Backward pass

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
|[![gd_no_vectorization.png](https://s19.postimg.cc/w0z9g6nib/gd_no_vectorization.png)](https://postimg.cc/image/shdbqdksf/)| [![gd_vectorization.png](https://s19.postimg.cc/pna6cxawj/gd_vectorization.png)](https://postimg.cc/image/u96al9wfj/)|
|Two for loops, one over the training examples \\(x^{(i)}\\) and a second over the features \\(x^{(i)}_j\\) . We have omitted the outermost loop that iterates over gradient steps. | Note that, we still need a single for loop to iterate over each gradient step (regardless if we are using stochastic or mini-batch gradient descent) even in our vectorized approach. |

#### Broadcasting

Lets motivate the usefulness of __broadcasting__ with an example. Lets say you wanted to get the percent of total calories from carbs, proteins, and fats for multiple foods.

[![food_matrix.png](https://s19.postimg.cc/q01kj1vqb/food_matrix.png)](https://postimg.cc/image/le5gapa73/)

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


__More broadcasting examples__

###### Addition

_Example 1_: \\(\begin{bmatrix}1 \\\\ 2 \\\\ 3 \\\\ 4\end{bmatrix} + 100 == \begin{bmatrix}1 \\\\ 2 \\\\ 3 \\\\ 4\end{bmatrix} + \begin{bmatrix}100 \\\\ 100 \\\\ 100 \\\\ 100\end{bmatrix} = \begin{bmatrix}101 \\\\ 102 \\\\ 103 \\\\ 104\end{bmatrix}\\)

_Example 2_: \\(\begin{bmatrix}1 & 2 & 3 \\\\ 4 & 5 & 6\end{bmatrix} + \begin{bmatrix}100 & 200 & 300\end{bmatrix} == \begin{bmatrix}1 & 2 & 3 \\\\ 4 & 5 & 6\end{bmatrix} + \begin{bmatrix}100 & 200 & 300 \\\\ 100 & 200 & 300\end{bmatrix} = \begin{bmatrix}101 & 202 & 303 \\\\ 104 & 205 & 306\end{bmatrix}\\)

_Example 3_: \\(\begin{bmatrix}1 & 2 & 3  \\\\ 4 & 5 & 6\end{bmatrix} + \begin{bmatrix}100 \\\\ 200\end{bmatrix} == \begin{bmatrix}1 & 2 & 3 \\\\ 4 & 5 & 6\end{bmatrix} + \begin{bmatrix}100 & 100 & 100 \\\\ 200 & 200 & 200\end{bmatrix} = \begin{bmatrix}101 & 202 & 303 \\\\ 104 & 205 & 206\end{bmatrix}\\)

#### (AISDE) A note on python/numpy vectors

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

## Week 3: Shallow neural networks

### Neural network overview

Up until this point, we have used logistic regression as a stand-in for neural networks. The "network" we have been describing looked like:

Network       | Computation Graph 	|
|:-------------------------:|:-------------------------:|
|[![lr_overview.png](https://s19.postimg.cc/3o3rppemr/lr_overview.png)](https://postimg.cc/image/jmchfu8un/)|[![lr_overview_graph.png](https://s19.postimg.cc/mgfmtblbn/lr_overview_graph.png)](https://postimg.cc/image/cj4m09dpr/)|


> \\(a\\) and  \\(\hat y\\) are used interchangeably

A neural network looks something like this:

Network       | Computation Graph 	|
|:-------------------------:|:-------------------------:|
|[![nn_overview.png](https://s19.postimg.cc/77ppfl9nn/nn_overview.png)](https://postimg.cc/image/c6d7u4dgf/)|[![nn_overview_graph.png](https://s19.postimg.cc/mt70ziygj/nn_overview_graph.png)](https://postimg.cc/image/535cehkvj/)|

> We typically don't distinguish between \\(z\\) and \\(a\\) when talking about neural networks, one neuron = one activation = one \\(a\\) like calculation.

We will introduce the notation of superscripting values with \\(^{[l]}\\), where \\(l\\) refers to the layer of the neural network that we are talking about.

> Not to be confused with \\(^{(i)}\\) which we use to refer to a single input example \\(i\\) .

_The key intuition is that neural networks stack activations of inputs multiplied by their weights_.

Similar to the 'backwards' step that we discussed for logistic regression, we will explore the backwards steps that makes learning in a neural network possible.

#### Neural network Representation

This is the canonical representation of a neural network

[![neural_network_basics.png](https://s19.postimg.cc/vbgh3vcoz/neural_network_basics.png)](https://postimg.cc/image/4qdy8babj/)

On the left, we have the __input features__ stacked vertically. This constitutes our __input layer__. The final layer, is called the __output layer__ and it is responsible for generating the predicted value \\(\hat y\\) . Any layer in between this two layers is known as a __hidden layer__. This name derives from the fact that the _true values_ of these hidden units is not observed in the training set.

> The hidden layers and output layers have parameters associated with them. These parameters are denoted \\(W^{[l]}\\) and \\(b^{[l]}\\) for layer \\(l\\) .

Previously, we were referring to our input examples as \\(x^{(i)}\\) and organizing them in a design matrix \\(X\\) . With neural networks, we will introduce the convention of denoting output values of a layer \\(l\\), as a column vector \\(a^{[l]}\\), where \\(a\\) stands for _activation_. You can also think of these as the values a layer \\(l\\) passes on to the next layer.

Another note: the network shown above is a _2-layer_ neural network. We typically do not count the input layer. In light of this, we usually denote the input layer as \\(l=0\\).

#### Computing a Neural Networks Output

We will use the example of a single hidden layer neural network to demonstrate the forward propagation of inputs through the network leading to the networks output.

We can think of each unit in the neural network as performing two steps, the _multiplication of inputs by weights and the addition of a bias_, and the _activation of the resulting value_

[![unit_breakdown.png](https://s19.postimg.cc/qquaof5oz/unit_breakdown.png)](https://postimg.cc/image/i8kuk2z67/)

> Recall, that we will use a superscript, \\(^{[l]}\\) to denote values belonging to the \\(l-th\\) layer.

So, the \\(j^{th}\\) node of the \\(l^{th}\\) layer performs the computation

\\[ a_j^{[l]} = \sigma(w_i^{[l]^T}a^{[l-1]} + b_i^{[l]}) \\]

> Where \\(a^{[l-1]}\\) is the activation values from the precious layer.

for some input \\(x\\). With this notation, we can draw our neural network as follows:

[![new_notation_nn.png](https://s19.postimg.cc/6i6x39r4j/new_notation_nn.png)](https://postimg.cc/image/f0gd7lxn3/)

In order to easily vectorize the computations we need to perform, we designate a matrix \\(W^{[l]}\\) for each layer \\(l\\), which has dimensions _(number of units in current layer X number of units in previous layer)_

We can vectorize the computation of \\(z^{[l]}\\) as follows:

[![vectorized_z_nn.png](https://s19.postimg.cc/n78cynd9v/vectorized_z_nn.png)](https://postimg.cc/image/66pgpz08f/)

And the computation of \\(a^{[l]}\\) just becomes the element-wise application of the sigmoid function:

[![vectorized_a_nn.png](https://s19.postimg.cc/7yifkuh0j/vectorized_a_nn.png)](https://postimg.cc/image/n78cymaov/)

We can put it all together for our two layer neural network, and outline all the computations using our new notation:

[![putting_it_all_together_new_notation.png](https://s19.postimg.cc/5so4qvgab/putting_it_all_together_new_notation.png)](https://postimg.cc/image/h50q8noz3/)

#### Vectorizing across multiple examples

In the last video, we saw how to compute the prediction for a neural network with a single input example. In this video, we introduce a vectorized approach to compute predictions for many input examples.  

We have seen how to take a single input example \\(x\\) and compute \\(a^{[2]} = \hat y\\) for a 2-layered neural network. If we have \\(m\\) training examples, we can used a vectorized approach to compute all \\(m\\) predictions.

First, lets introduce a new notation. The activation values of layer \\(l\\) for input example \\(i\\) is:

\\[ a^{[l](i)} \\]

The \\(m\\) predictions our 2-layered are therefore computed in the following way:

[![m_examples_nn.png](https://s19.postimg.cc/mt70zhvvn/m_examples_nn.png)](https://postimg.cc/image/i7awr5acf/)

Recall that \\(X\\) is a \\((n_x, m)\\) design matrix, where each column is a single input example and \\(W^{[l]}\\) is a matrix where each row is the transpose of the parameter column vector for layer \\(l\\).

Thus, we can now compute the activation of a layer in the neural network for all training examples:

\\[Z^{[l]} = W^{[l]}X + b^{[l]}\\]
\\[A^{[l]} = sign(Z^{[l]})\\]

As an example, the result of a matrix multiplication of \\(W^{[1]}\\) by \\(X\\) is a matrix with dimensions \\((j, m)\\) where \\(j\\) is the number of units in layer \\(1\\) and \\(m\\) is the number of input examples

[![WX_vector.jpg](https://s19.postimg.cc/6w892blcj/WX_vector.jpg)](https://postimg.cc/image/un7mkfljj/)

\\(A^{[l]}\\) is therefore a matrix of dimensions (size of layer \\(l\\) X \\(m\\)). The top-leftmost value is the activation for the first unit in the layer \\(l\\) for the first input example \\(i\\), and the bottom-rightmost value is the activation for the last unit in the layer \\(l\\) for the last input example \\(m\\) .

[![vectorized_activations.png](https://s19.postimg.cc/dmoqbqt2r/vectorized_activations.png)](https://postimg.cc/image/o9ijh617z/)

### Activation Functions

So far, we have been using the __sigmoid__ activation function

\\[\sigma(z) = \frac{1}{1 + e^{-z}}\\]

It turns out there are much better options.

#### Tanh

The __hyperbolic tangent function__ is a non-linear activation function that almost always works better than the sigmoid function.

\\[tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}\\]

![](https://upload.wikimedia.org/wikipedia/commons/thumb/7/76/Sinh_cosh_tanh.svg/640px-Sinh_cosh_tanh.svg.png?1514655794955.png)

> The tanh function is really just a shift of the sigmoid function so that it crosses through the origin.

The tanh activation usually works better than sigmoid activation function for hidden units because the mean of its output is closer to zero, and so it centers the data better for the next layer.

The single exception of sigmoid outperforming tanh is when its used in the ouput layer. In this case, it can be more desirable to scale our outputs from \\(0\\) to \\(1\\) (particularly in classification, when we want to output the probability that something belongs to a certain class). Indeed, we often mix activation functions in neural networks, and denote them:

\\[g^{[p]}(z)\\]

Where \\(p\\) is the \\(p^th\\) activation function.

If \\(z\\) is either very large, or very small, the derivative of both the tanh and sigmoid functions becomes very small, and this can slow down learning.

#### ReLu

The __rectified linear unit__ activation function solves the disappearing gradient problem faced by tanh and sigmoid activation functions. In practice, it also leads to faster learning.

\\[ReLu(z) = max(0, z)\\]

![](https://upload.wikimedia.org/wikipedia/commons/thumb/6/6c/Rectifier_and_softplus_functions.svg/640px-Rectifier_and_softplus_functions.svg.png?1528644452536)

> Note: the derivative at exactly 0 is not well-defined. In practice, we can simply set it to 0 or 1 (it matters little, due to the unlikeliness of a floating point number to ever be \\(0.0000...\\) exactly).

One disadvantage of ReLu is that the derivative is equal to \\(0\\) when \\(z\\) is negative. __Leaky ReLu__'s aim to solve this problem with a slight negative slope for values of \\(z<0\\) .

\\[ReLu(z) = max(0.01 * z, z)\\]

![](http://lamda.nju.edu.cn/weixs/project/CNNTricks/imgs/leaky.png)

> Image sourced from [here](http://lamda.nju.edu.cn/weixs/project/CNNTricks/imgs/leaky.png).

Sometimes, the \\(0.01\\) value is treated as an adaptive parameter of the learning algorithm. Leaky ReLu's solve a more general problem of "[dead neurons](https://www.quora.com/What-is-the-definition-of-a-dead-neuron-in-Artificial-Neural-Networks?share=1)". However, it is not used as much in practice.

__Rules of thumb for choosing activations functions__

- _If your output is a 0/1 value_, i.e., you are performing binary classification, the sigmoid activation is a natural choice for the output layer.
- _For all other units_, ReLu's is increasingly the default choice of activation function.

__Why do you need non-linear activation functions?__

We could imagine using some __linear__ activation function, \\(g(z) = z\\) in place of the __non-linear__ activation functions we have been using so far. Why is this a bad idea? Lets illustrate out explanation using our simple neural networks

[![neural_network_basics.png](https://s19.postimg.cc/vbgh3vcoz/neural_network_basics.png)](https://postimg.cc/image/4qdy8babj/)

For this linear activation function, the activations of our simple network become:

\\[z^{[1]} = W^{[1]}x + b^{[1]}\\]
\\[a^{[1]} = z^{[1]}\\]
\\[z^{[2]} = W^{[2]}x + b^{[2]}\\]
\\[a^{[2]} = z^{[2]}\\]

From which we can show that,

\\[a^{[2]} = (W^{[2]}W^{[1]})x + (W^{[2]}b^{[1]} + b^{[2]})\\]
\\[a^{[2]} = W'x + b' \text{, where } W' = W^{[2]}W^{[1]} \text{ and } b' = W^{[2]}b^{[1]} + b^{[2]}\\]

Therefore, in the case of a _linear activation function_, the neural network is outputting a _linear function of the inputs_, no matter how many hidden layers!

###### Exceptions

There are (maybe) two cases in which you may actually want to use a linear activation function.

1. The output layer of a network used to perform regression, where we want \\(\hat y\\) to be a real-valued number, \\(\hat y \in \mathbb R\\)
2. Extremely specific cases pertaining to compression.

#### Derivatives of activation functions

When perform back-propogation on a network, we need to compute the derivatives of the activation functions. Lets take a look at our activation functions and their derivatives

__Sigmoid__

[![sigmoid_deriv.png](https://s19.postimg.cc/dy66p1jyr/sigmoid_deriv.png)](https://postimg.cc/image/535ceiv67/)

The deriviative of \\(g(z)\\), \\(g(z)'\\) is:

\\[\frac{d}{dz}g(z) = \frac{1}{1 + e^{-z}}(1 - \frac{1}{1 + e^{-z}})= g(z)(1-g(z)) = a(1-a)\\]

> We can sanity check this by inputting very large, or very small values of \\(z\\) into our derivative formula and inspecting the size of the outputs.

Notice that if we have already computed the value of \\(a\\), we can very cheaply compute the value of \\(g(z)'\\) .

__Tanh__

[![tanh_deriv.png](https://s19.postimg.cc/g2qjq510z/tanh_deriv.png)](https://postimg.cc/image/i7awr82nj/)

The deriviative of \\(g(z)\\), \\(g(z)'\\) is:

\\[\frac{d}{dz}g(z) = 1 - (tanh(z))^z\\]

> Again, we can sanity check this inspecting that the outputs for different values of \\(z\\) match our intuition about the activation function.

__ReLu__

[![relu_deriv.png](https://s19.postimg.cc/i7awr6scz/relu_deriv.png)](https://postimg.cc/image/iwtp3jswf/)

The derivative of \\(g(z)\\), \\(g(z)'\\) is:

\\[\frac{d}{dz}g(z) = 0 \text{ if } z < 0 ; 1 \text{ if } z > 0; \text{ undefined if } z = 0\\]

> If \\(z = 0\\), we typically default to setting \\(g(z)\\) to either \\(0\\) or \\(1\\) . In practice this matters little.

### Gradient descent for Neural Networks

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

#### Formulas for computing derivatives

We are going to simply present the formulas you need, and defer their explanation to the next video. Recall the computation graph for our 2-layered neural network:

[![nn_overview_graph.png](https://s19.postimg.cc/mt70ziygj/nn_overview_graph.png)](https://postimg.cc/image/535cehkvj/)|

And the vectorized implementation of our computations in our __forward propagation__

1.\\[Z^{[1]} = W^{[1]}X + b^{[1]}\\]
2.\\[A^{[1]} = g^{[1]}(z^{[1]})\\]
3.\\[Z^{[2]} = W^{[2]}X + b^{[2]}\\]
4.\\ß[A^{[2]} = g^{[2]}(z^{[2]})\\]

> Where \\(g^{[2]}\\) would likely be the sigmoid function if we are doing binary classification.

Now we list the computations for our __backward propagation__

1.\\[ dZ^{[2]} = A^{[2]} - Y \\]
2.\\[ dW^{[2]} = \frac{1}{m}dZ^{[2]}A^{[1]T} \\]

> Transpose of A accounts for the fact that W is composed of transposed column vectors of parameters.

3.\\[db^{[2]} = \frac{1}{m}np.sum(dZ^{[2]}, axis = 1, keepdims=True)\\]

> Where \\(Y = [y^{(1)}, ..., y^{[m]}]\\) . The `keepdims` arguments prevents numpy from returning a rank 1 array, \\((n,)\\)

4.\\[dZ^{[1]} = W^{[2]T}dZ^{[2]} \odot g(Z)' (Z^{[1]})\\]

> Where \\(\odot\\) is the element-wise product. Note: this is a collapse of \\(dZ\\) and \\(dA\\) computations.

5.\\[dW{[1]} = \frac{1}{m} = dZ^{[1]}X^T\\]
6.\\[db^{[1]} = \frac{1}{m}np.sum(dZ^{[1]}, axis=1, keepdims=True)\\]

### Random Initialization

When you train your neural network, it is important to initialize your parameters _randomly_. With logistic regression, we were able to initialize our weights to _zero_ because the cost function was convex. We will see that this _will not work_ with neural networks.

Lets take the following network as example:

[![super_simple_network.png](https://s19.postimg.cc/8b9tr0jur/super_simple_network.png)](https://postimg.cc/image/aslkya3r3/)

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

__Why do we initialize to small values?__

For a _sigmoid-like_ activation function, large parameter weights (positive or negative) will make it more likely that \\(z\\) is very large (positive or negative) and thus \\(dz\\) will approach \\(0\\), _slowing down learning dramatically_.

> Note this is a less of an issue when using ReLu's, however many classification problems use sigmoid activations in their output layer.

## Week 4: Deep Neural Networks

### What is a deep neural network?

A deep neural network is simply a network with more than 1 hidden layer. Compared to logistic regression or a simple neural network with one hidden layer (which are considered __shallow__ models) we say that a neural network with many hidden layers is a __deep__ model, hence the terms _deep learning / deep neural networks_.

> Shallow Vs. deep is a matter of degree, the more hidden layers, the deeper the model.

![](https://s19.postimg.org/ku99a0jmb/shallow_vs_deep.png)

Over the years, the machine learning and AI community has realized that deep networks are excellent function approximators, and are able to learn incredibly complex functions to map inputs to outputs.

> Note that is difficult to know in advance how deep a neural network needs to be to learn an effective mapping.

#### Notation

Lets go over the notation we will need using an example network

![](https://s19.postimg.org/b9pmn6rqb/simple_deep_nn.png)

- We will use \\(L\\) to denote the number of layers in the network (in this network \\(L=4\\))
- \\(n^{[l]}\\) denotes the number of units in layers \\(l\\) (for example, in this network \\(n^{[1]} = 5\\))
- \\(a^{[l]}\\) denotes the activations in layer \\(l\\)
- \\(W^{[l]}\\), \\(b^{[l]}\\) denotes the weights and biases for \\(z^{[l]}\\)
- \\(x = a^{[0]}\\) and \\(\hat y = a^{[L]}\\)

### Forward Propagation in a Deep Network

Forward propogation in a deep network just extends what we have already seen for forward propogation in a neural network by some number of layers. More specifically, for each layer \\(l\\) we perform the computations:

\\[ Z^{[l]} = W^{[l]}A^{[l-1]} + b^{[l]} \\]
\\[ A^{[l]} = g^{[l]}(Z^{[l]}) \\]

> Note that the above implementation is vectorized across all training examples. Matrices \\(A^{[l]}\\) and \\(Z^{[l]}\\) stacked column vectors pertaining to a single input example for layer \\(l\\).

Finally, our predictions (the results of our output layer) are:

\\[\hat Y = g(Z^{[L]}) = A^{[L]}\\]

> Note that this solution is not completely vectorized, we still need an explicit for loop over our layers \\(l = 0, 1, ..., L\\)

#### Getting your matrix dimensions right

When implementing a neural network, it is extremely important that we ensure our matrix dimensions "line up". A simple debugging tool for neural networks then, is pen and paper!

For a \\(l\\)-layered neural network, our dimensions are as follows:

- \\(W^{[l]}: (n^{[l]}, n^{[l-1]})\\)
- \\(b^{[l]}: (n^{[l]}, 1)\\)
- \\(Z^{[l]}, A^{[l]}: (n^{[l]}, m)\\)
- \\(A^{[0]} = X: (n^{[0]}, m)\\)

Where \\(n^{[l]}\\) is the number of units in layer \\(l\\).

> See [this](https://www.coursera.org/learn/neural-networks-deep-learning/lecture/rz9xJ/why-deep-representations) video for a derivation of these dimensions.

When implementing backpropagation, the dimensions are the same, i.e., the dimensions of \\(W\\), \\(b\\), \\(A\\) and \\(Z\\) are the same as \\(dW\\), \\(db\\), ...

### Why deep representations?

Lets train to gain some intuition behind the success of deep representation for certain problem domains.

#### What is a deep network computing?

Lets take the example of image recognition. Perhaps you input a picture of a face, then you can think of the first layer of the neural network as an "edge detector".

The next layer can use the outputs from the previous layer, which can roughly be thought of as detected edges, and "group" them in order to detect parts of faces. Each neuron may become tuned to detect different parts of faces.

Finally, the output layer uses the output of the previous layer, detected features of a face, and compose them together to recognize a whole face.

[![deep_representations.png](https://s19.postimg.cc/57hzx2doj/deep_representations.png)](https://postimg.cc/image/slpz8zvlr/)

> The main intuition is that earlier layers detect "simpler" structures, and pass this information onto the next layer which can use it to detect increasingly complex structures.

These general idea applies to other examples than just computer vision tasks (e.g., audio). Moreover, there is an analogy between deep representations in neural networks and how the brain works, however it can be dangerous to push these analogies too far.

#### Circuit theory and deep learning

Circuit theory also provides us with a possible explanation as to why deep networks work so well for some tasks. Informally, there are function you can compute with a "small" L-layer deep neural network that shallower networks require exponentially more hidden units to compute.

> Check out [this](https://www.coursera.org/learn/neural-networks-deep-learning/lecture/rz9xJ/why-deep-representations) video starting at 5:36 for a deeper explanation of this.

### Building blocks of deep neural networks

Lets take a more holistic approach and talk about all the building blocks of deep neural networks. Here is a deep neural network with a few hidden layers

[![simple_deep_nn_2.png](https://s19.postimg.cc/wtuvboc5v/simple_deep_nn_2.png)](https://postimg.cc/image/fgkkwtgun/)

Lets pick one layer, \\(l\\) and look at the computations involved.

For this layer \\(l\\), we have parameters \\(W^{[l]}\\) and \\(b^{[l]}\\). Our two major computation steps through this layer are:

__Forward Propagation__

- Input: \\(a^{[l-1]}\\)
- Output: \\(a^{[l]}\\)
- Linear function: \\(z^{[l]} = W^{[l]}a^{[l-1] + b^{[l]}}\\)
- Activation function: \\(a^{[l]} = g^{[l]}(z^{[l]})\\)

Because \\(z^{[l]}, W^{[l]} and b^{[l]}\\) are used in then backpropagation steps, it helps to cache theses values during forward propagation.

__Backwards Propagation__

- Input: \\(da^{[l]}, cache(z^{[l]})\\)
- Output: \\(da^{[l-1]}, dW^{[l]}, db^{[l]}\\)

The key insight, is that for every computation in forward propagation there is a corresponding computation in backwards propagation

[![forward_backward.png](https://s19.postimg.cc/nfx5yyrtf/forward_backward.png)](https://postimg.cc/image/ct3ctjjnz/)

So one iteration of training with a neural network involves feeding our inputs into the network (\\(a^{[0]})\\), performing forward propagation computing \\(\hat y\\), and using it to compute the loss and perform backpropagation through the network. This will produce all the derivatives of the parameters w.r.t the loss that we need to update the parameters for gradient descent.

### Parameters vs hyperparameters

The __parameters__ of your model are the _adaptive_ values, \\(W\\) and \\(b\\) which are _learned_ during training via gradient descent.

In contrast, __hyperparameters__ are set before training and can be viewed as the "settings" of the learning algorithms. They have a direct effect on the eventual value of the parameters.

Examples include:

- number of iterations
- learning rate
- number of hidden layers \\(L\\)
- number of hidden units \\(n^{[1]}, n^{[2]}, ...\\)
- choice of activation function

> the learning rate is sometimes called a parameter. We will follow the convetion of calling it a hyperparameter.

It can be difficult to know the optimal hyperparameters in advance. Often, we start by simply trying out many values to see what works best, this allows us to build our intuition about the best hyperparameters to use. We will defer a deep discussion on how to choose hyperparameters to the next course.

### What does this all have to do with the brain?

At the risk of giving away the punch line, _not a whole lot_.

The most important mathematical components of a neural networks: _forward propagation_ and _backwards propagation_ are rather complex, and it has been difficult to convey the intuition behind these methods. As a result, the phrase, "it's like the brain" has become an easy, but dramatically oversimplified explanation. It also helps that this explanation has caught the publics imagination.  

There is a loose analogy to be drawn from a biological neuron and the neurons in our artificial neural networks. Both take inputs (derived from other neurons) process the information and propagate a signal forward.

![]()

However, even today neuroscientists don't fully understand what a neuron is doing when it receives and propagates a signal. Indeed, we have no idea on whether the biological brain is performing some algorithmic processes similar to those performed by an ANN.

Deep learning is an excellent method for complex function approximation, i.e., learning mappings from inputs \\(x\\) to outputs \\(y\\). However we should be very wary about pushing the, "its like a brain!" analogy too far.

# Course 3: Structuring Machine Learning Projects

#### TOC

1. [Week 1: ML Strategy 1](#ml-strategy-1)
2. [Week 1: ML Strategy 2](#ml-strategy-2)

## Week 1: ML Strategy (1)

What is _machine learning strategy?_ Lets start with a motivating example.

### Introduction to ML strategy

#### Why ML strategy

Lets say you are working on a __cat classifier__. You have achieved 90% accuracy, but would like to improve performance even further. Your ideas for achieveing this are:

- collect more data
- collect more diverse training set
- train the algorithm longer with gradient descent
- try adam (or other optimizers) instead of gradient descent
- try dropout, add L2 regularization, change network architecture, ...

This list is long, and so it becomes incredibly important to be able to identify ideas that are worth our time, and which ones we can likely discard.

This course will attempt to introduce a framework for making these decisions. In particular, we will focus on the organization of _deep learning-based projects_.

#### Orthogonalization

One of the challenges with building deep learning systems is the number of things we can tune to improve performance (_many hyperparameters notwithstanding_).

Take the example of an old TV. They included many nobs for tuning the display position (x-axis position, y-axis position, rotation, etc...).

__Orthogonalization__ in this example refers to the TV designers decision to ensure each nob had one effect on the display and that these effects were _relative_ to one another. If these nobs did more than one action and each actions magnitude was not relative to the other, it would become nearly impossible to tune the TV.

Take another example, driving a __car__. Imagine if there was multiple joysticks. One joystick modified \\(0.3\\) X steering angle \\(- 0.8\\) speed, and another \\(2\\) X steering angle \\(+ 0.9\\) speed. In theory, by tuning these two nobs we could drive the car, but this would be _much more difficult then separating the inputs into distinct input mechanisms_.

__Orthogonal__ refers to the idea that the _inputs_ are aligned to the dimensions we want to control.

[![speed_v_angle_orth.png](https://s19.postimg.cc/51dg3e5v7/speed_v_angle_orth.png)](https://postimg.cc/image/t547roobz/)

_How does this related to machine learning?_

##### Chain of assumption in examples

For a machine learning system to perform "well", we usually aim to make four things happen:

1. Fit training set well on cost function (for some applications, this means comparing favorably to human-level performance).
2. Fit dev set well on cost function
3. Fit test set well on cost function
4. Performs well in real world.

If we relate back to the TV example, we wanted _one knob_ to change each attribute of the display. _In the same way, we can modify knobs for each of our four steps above_:

1. Train a bigger network, change the optimization algorithm, ...
2. Regularization, bigger training set, ...
3. Bigger dev set, ...
4. Change the dev set or the cost function

> Andrew said when he trains neural networks, he tends __not__ to use __early stopping__. The reason being is that this is not a very __orthogonal__ "knob"; it simultaneously effects how well we fit the training set and the dev set.

The whole idea here is that if we keep our "knobs" __orthogonal__, we can more easily come up with solutions to specific problems with our deep neural networks (i.e., if we are getting poor performance on the training set, we may opt to train a bigger [higher variance] network).

### Setting up your goal

#### Single number evaluation metric

When tuning neural networks (modifying hyper-parameters, trying different architectures, etc.) you will find that having a _single __evaluation metric___ will allow you to easily and quickly judge if a certain change improved performance.

> Andrew recommends deciding on a single, real-valued evaluation metric when starting out on your deep learning project.

Lets look at an example.

As we discussed previously, __applied machine learning__ is a very empirical process.

[![using_a_single_number.png](https://s19.postimg.cc/3mbveorxf/using_a_single_number.png)](https://postimg.cc/image/6t6eybcdb/)

Lets say that we start with classifier A, and end up with classifier B after some change to the model. We could look at __precision__ and __recall__ as a means of improvements. What we really want is to improve _both_ precision and recall. The problem is that it can become difficult to choose the "best" classifier if we are monitoring two different performance metrics, especially when we are making many modifications to our network.

This is when it becomes important to chose a single performance metric. In this case specifically, we can chose the __F1-score__, the harmonic mean of the precision and recall (less formally, think of this as an average).

[![chosing_f1_score.png](https://s19.postimg.cc/ovzhpj0ib/chosing_f1_score.png)](https://postimg.cc/image/uwx6mln4f/)

We can see very quickly that classifier A has a better F1-score, and therefore we chose classifier A over classifier B.

#### Satisficing and Optimizing metric

It is not always easy to combine all the metrics we care about into a single real-numbered value. Lets introduce __satisficing__ and __optimizing__ metrics as a solution to this problem.

Lets say we are building a classifier, and we care about both our __accuracy__ (measured as F1-score, traditional accuracy or some other metric) _and_ the __running time__ to classify a new example.

[![two_metrics_optimize.png](https://s19.postimg.cc/px8x67gur/two_metrics_optimize.png)](https://postimg.cc/image/aoizsfn67/)

One thing we can do, is to combine accuracy and run-time into a __single-metric__, possibly by taking a weighted linear sum of the two metrics.

> As it turns out, this tends to produce a rather artificial solution (no pun intended).

Another way, is to attempt to _maximize accuracy_ while subject to the restraint that \\(\text{running time} \le 100\\)ms. In this case, we say that _accuracy_ is an __optimizing__ metric (because we want to maximize or minimize it) and _running time_ is a __satisficing__ metric (because it just needs to meet a certain constraint, i.e., be "good enough").

More generally, if we have \\(m\\) metrics that we care about, it is reasonable to choose _one_ to be our __optimizing metric__, and \\(m-1\\) to be __satisficing metrics__.

##### Example: Wake words

We can take a concrete example to illustrate this: __wake words__ for __intelligent voice assistants__. We might chose the accuracy of the model (i.e., what percent of the time does it "wake" when a wake word is said) to be out __optimizing metric__ s.t. we have \\(\le 1\\) false-positives per 24 hours of operation (our __satisficing metric__).

##### Summary

To summarize, if there are multiple things you care about, we can set one as the __optimizing metric__ that you want to do as well as possible on and one or more as __satisficing metrics__ were you'll be satisfied. This idea goes hand-in-hand with the idea of having a single real-valued performance metric whereby we can _quickly_ and _easily_ chose the best model given a selection of models.

### Train/dev/test distributions

The way you set up your train, dev (sometimes called valid) and test sets can have a large impact on your development times and even model performance.

In this video, we are going to focus on the __dev__ (sometimes called the __valid__ or __hold out__ set) and the __test set__. The general workflow in machine learning is to train on the __train__ set and test out model performance (e.g., different hyper-parameters or model architectures) on the __dev__ set.

Lets look at an example. Say we had data from multiple regions:

- US
- UK
- Other European countries
- South America
- India
- China
- Other Asian countries
- Australia

If we were to build our dev set by choosing data from the first four regions and our test set from the last four regions, our data would likely be __skewed__ and our model would likely perform poorly (at least on the __test__ set). _Why?_

Imagine the __dev__ set as a target, and our job as machine learning engineers is to hit a bullseye. _A dev set that is not representative of the overall general distribution is analogous to moving the bullseye away from its original location moments after we fire our bow_. An ML team could spend months optimizing the model on a dev set, only to achieve very poor performance on a test set!

So for our data above, a much better idea would be to sample data randomly from all regions to build our __dev__ and __test__ set.

#### Guidelines

Choose a __dev__ set and __test__ set (from the same distribution) to reflect data you expect to _get in the future_ and _consider important to do well on_.

### Size of the dev and test sets

In the last lecture we saw that the dev and test sets should come from the same distributions. _But how large should they be?_

#### Size of the dev/test sets

The rule of thumb in machine learning is typically 60% __training__, 20% __dev__, and 20% __test__ (or 70/30 __train__/__test__). In earlier eras of machine learning, this was pretty reasonable. In the modern machine learning era, we are used to working with _much_ larger data set sizes.

For example, imagine we have \\(1,000,000\\) examples. It might be totally reasonable for us to use 98% as our test set, 1% for dev and 1% for __test__.

> Note that 1% of \\(10^6\\) is \\(10^4\\)!

#### Guidelines

Set your __test__ set to be big enough to give high confidence in the overall performance of your system.

### When to change dev/test sets and metrics

Sometimes during the course of a machine learning project, you will realize that you want to change your evaluation metric (i.e., move the "goal posts"). Lets illustrate this with an example:

#### Example 1

Imagine we have two models for image classification, and we are using classification performance as our evaluation metric:

- Algorithm A has a __3%__ error, but sometimes shows users pornographic images.
- Algorithm B has a __5%__ error.

Cleary, algorithm A performs better by our original evaluation metric (classification performance), but showing users pornographic images is _unacceptable_.

\\[Error = \frac{1}{m_{dev}}\sum^{m_{dev}}_{i=1} \ell \{ y_{pred}^{(i)} \ne y^{(i)} \}\\]

> Our error treats all incorrect predictions the same, pornographic or otherwise.

We can think of it like this: our evaluation metric _prefers_ algorithm A, but _we_ (and our users) prefer algorithm B. When our evaluation metric is no longer ranking the algorithms in the order we would like, it is a sign that we may want to change our evaluation metric. In our specific example, we could solve this by weighting misclassifications

\\[Error = \frac{1}{w^{(i)}}\sum^{m_{dev}}_{i=1} w^{(i)}\ell \{ y_{pred}^{(i)} \ne y^{(i)} \}\\]

where \\(w^{(i)}\\) is 1 if \\(x^{(i)}\\) is non-porn and 10 (or even 100 or larger) if \\(x^{(i)}\\) is porn.

This is actually an example of __orthogonalization__. We,

1. Define a metric to evaluate our model ("placing the target")
2. (In a completely separate step) Worry about how to do well on this metric.

#### Example 2

Take the same example as above, but with a new twist. Say we train our classifier on a data set of high quality images. Then, when we deploy our model we notice it performs poorly. We narrow the problem down to the low quality images users are "feeding" to the model. What do we do?

In general: _if doing well on your metric + dev/test set does not correspond to doing well on your application, change your metric and/or dev/test set_.

### Comparing to human-level performance

In the last few years, comparing machine learning systems to human level performance have become common place. The reasons for this include:

1. Deep learning based approaches are making extraordinary gains in performance, so our baseline needs to be more stringent.
2. Many of the tasks deep learning is performing well at were thought to be very difficult for machines (e.g. NLP, computer vision). Comparing performance on these tasks to a human baseline is natural.

It is also instructive to look at the performance of machine learning over time (note this is an obvious abstraction)

[![ai_progress_over_time.png](https://s19.postimg.cc/8ij85de7n/ai_progress_over_time.png)](https://postimg.cc/image/wm9ztnwof/)

Roughly speaking, performance (e.g., in a research domain or for a certain task) progresses quickly until we reach human-level performance, and tails off quickly. _Why?_ mainly because human level performance is typically very close to the [__Bayes optimal error__](http://www.wikiwand.com/en/Bayes_error_rate). Bayes optimal error is the best possible error; there is no way for any function mapping from \\(x \rightarrow y\\) to do any better. A second reason is that so long as ML performs worse than humans for a given task, we can:

- get labeled data from humans
- gain insight from manual error analysis (e.g., why did a person get this right?)
- better analysis of bias/variance

### Avoidable bias

Of course, we want our learning algorithm to perform well on the training set, but not _too well_. Knowing where human level performance is can help us decide how well we want to perform on the training set.

Let us again take the example of an image classifier. For this particular data set, assume:

- human-level performance is an error of 1%.
- our classifier is currently achieving 8% classification error on the training set and
- 10% classification on the dev set.

_Clearly, it has plenty of room to improve_. Specifically, we would want to try to _increase_ **variance** and _reduce_ __bias__.

> For the purposes of computer vision, assume that human-level performance \\(\approx\\) Bayes error.

Now, lets take the same example, but instead, we assume that human-level performance is an error of 7.5% (this example is very contrived, as humans are extremely good at image classification). In this case, we note that our classifier performances nearly as well as a human baseline. We would likely want to to _decrease_ **variance** and _increase_ __bias__ (in order to improve performance on the __dev__ set.)

So what did this example show us? When human-level performance (where we are using human-level performance as a proxy for Bayes error) is _very high_ relative to our models performance on the train set, we likely want to focus on reducing  _"avoidable"_ bias (or increasing variance) in order to improve performance on the training set (e.g., by using a bigger network.) When human-level performance is _comparable_ to our models performance on the train set, we likely want to focus on increasing bias (or decreasing variance) in order to improve performance on the dev set (e.g., by using a regularization technique or gathering more training data.)

### Understanding human-level performance

The term _human-level performance_ is used quite casually in many research articles. Lets attempt to define this term more precisely.

Recall from the last lecture that **human-level performance** can be used as a proxy for **Bayes error**. Lets revisit that idea with another example.

Suppose, for a medical image classification example,

- Typical human: 3% error
- Typical doctor: 1% error
- Experienced doctor: 0.7% error
- Team of experienced doctors: 0.5% error

_What is "human-level" error?_ Most likely, we would say __0.5%__, and thus Bayes error is \\(\le 0.05%\\).  However, in certain contexts we may only wish to perform as well as the typical doctor (i.e., 1% error) and we may deem this _"human-level error"_. The takeaway is that there is sometimes more than one way to determine human-level performance; which way is appropriate will depend on the context in which we expect our algorithm to be deployed. We also note that as the performance of our algorithm improves, we may decide to move the goal posts for human-level performance higher, e.g., in this example by choosing a team of experienced doctors as the baseline. This is useful for solving the problem introduced in the previous lecture: _should I focus on reducing avoidable bias? or should I focus on reducing variance between by training and dev errors._

#### Summary

Lets summarize: if you are trying to understand bias and variance when you have a human-level performance baseline:

- Human-level error can be used as a proxy for Bayes' error
- The difference between the training error and the human-level error can be thought of as the __avoidable bias__.
- The difference between the training and dev errors can be thought of as __variance__.
- Which type of error you should focus on reducing depends on how well your model perform compares to (an estimate of) human-level error.
- As our model approaches human-level performance, it becomes harder to determine where we should focus our efforts.

### Surpassing human-level performance

Surpassing human-level performance is what many teams in machine learning / deep learning are inevitably trying to do. Lets take a look at a harder example to further develop our intuition for an approach to _matching_ or _surpassing_ human-level performance.

- team of humans: 0.5% error
- one human: 1.0% error
- training error: 0.3% error
- dev error: 0.4% error

Notice that training error < team of humans error. Does this mean we have _overfit_ the data by 0.2%? Or, does this means Bayes' error is actually lower than the team of humans error? We don't really know based on the information given, as to whether we should focus on __bias__ or __variance__. This example is meant to illustrate that once we surpass human-level performance, it becomes much less clear how to improve performance further.

#### Problems where ML significantly surpasses human-level performance

Some example where ML _significantly surpasses human-level performance_ include:

- Online advertising,
- Product recommendations
- Logistics (predicting transit time)
- Load approvals

Notice that many of these tasks are learned on __structured data__ and do not involve __natural perception tasks__. This appeals to our intuition, as we know humans are _excellent_ at natural perception tasks.

> We also note that these four tasks have immensely large datasets for learning.

### Improving your model performance

You have heard about orthogonalization. How to set up your dev and test sets, human level performance as a proxy for Bayes's error and how to estimate your avoidable bias and variance. Let's pull it all together into a set of guidelines for how to improve the performance of your learning algorithm.

#### The two fundamental assumptions of supervised learning

1. You can fit the training set (pretty) well, i.e., we can achieve _low avoidable bias_.
2. The training set performance generalizes pretty well to the dev/test set, i.e., variance is _not too bad_.

In the spirit of orthogonalization, there are a certain set of (separate) knobs we can use to improve bias and variance. Often, the difference between the training error and Bayes error (or a human-level proxy) is often illuminating in terms of where large improvement remain to be made.

_For reducing bias_

- Train a bigger model
- Train longer/better optimization algorithms
- Change/tweak NN architecture/hyperparameter search.

_For reducing variance_

- Collect more data
- Regularization (L2, dropout, data augmentation)
- Change/tweak NN architecture/hyperparameter search.

## Week 2: ML Strategy (2)

### Error Analysis

Manually examining mistakes that your algorithm is making can give you insights into what to do next (_especially if your learning algorithm is not yet at the performance of a human_). This process is called **error analysis**. Let's start with an example.

#### Carrying out error analysis

Take for example our __cat image classifier__, and say we obtain 10% error on our **test set**, much worse than we were hoping to do. Assume further that a colleague notices some of the misclassified examples are actually pictures of dogs. The question becomes, _should you try to make your cat classifier do better on dogs?_

This is where **error analysis** is particularly useful. In this example, we might:

- collect ~100 mislabeled dev set examples
- count up how any many dogs

Lets say we find that 5/100 (5%) mislabeled dev set example are dogs. Thus, the best we could hope to do (if we were to _completely_ solve the dog problem) is decrease our error from 10% to 9.5% (a 5% relative drop in error.) We conclude that _this is likely not the best use of our time_. Sometimes, this is called the __ceiling__, i.e., the _maximum_ amount of improvement we can expect from _some change_ to our algorithm/dataset.

Suppose instead we find 50/500 (50%) mislabeled dev set examples are dogs. Thus, if we solve the dog problem, we could decrease our error from 10% to 5% (a 50% relative drop in error.) We conclude that _solving the dog problem is likely a good use of our time_.

> Notice the disproportionate 'payoff' here. It may take < 10 min to manually examine 100 examples from our dev set, but the exercise offers _major_ clues as to where to focus our efforts.

##### Evaluate multiple ideas in parallel

Lets, continue with our cat detection example. Sometimes we might want to evaluate **multiple** ideas in __parallel__. For example, say we have the following ideas:

- fix pictures of dogs being recognized as cats
- fix great cats (lions, panthers, etc..) being misrecognized
- improve performance on blurry images

What can do is create a table, where the _rows_ represent the images we plan on evaluating manually, and the _columns_ represent the categorizes we think the algorithm may be misrecognizing. It is also helpful to add comments describing the the misclassified example.

![](https://s19.postimg.org/thwsxyhrn/Screen_Shot_2018-02-24_at_9.39.51_AM.png)

As you are part-way through this process, you may also notice another common category of mistake, which you can add to this manual evaluation and repeat.

_The conclusion of this process is estimates for:_

- which errors we should direct our attention to solving
- how much we should expect performance to improve if reduce the number of errors in each category

##### Summary

To summarize: when carrying out error analysis, you should find a set of _mislabeled_ examples and look at these examples for _false positives_ and _false negatives_. Counting up the number of errors that fall into various different categories will often this will help you prioritize, or give you inspiration for new directions to go in for improving your algorithm.

Three numbers to keep your eye on

1. Overall dev set error
2. Errors due to cause of interest / Overall dev set error
3. Error due to other causes / Overall dev set error

If the errors due to other causes >> errors due to cause of interest, it will likely be more productive to ignore our cause of interest for the time being and seek another source of error we can try to minimize.

> In this case, _cause of interest_ is just our idea for improving our leaning algorithm, e.g., _fix pictures of dogs being recognized as cats_

### Cleaning up incorrectly labeled data

In supervised learning, we (typically) have hand-labeled training data. What if we realize that some examples are _incorrectly labeled?_ First, lets consider our training set.

> In an effort to be less ambiguous, we use __mislabeled__ when we are referring to examples the ML algo labeled incorrectly and **incorrectly** labeled when we are referring to examples in the training data set with the wrong label.

#### Training set

Deep learning algorithms are quite robust to **random** errors in the training set. If the errors are reasonably **random** and the dataset is big enough (i.e., the errors make up only a tiny proportion of all examples) performance of our algorithm is unlikely to be affected.

**Systematic errors** are much more of a problem. Taking as example our cat classifier again, if labelers mistakingly label all white dogs as cats, this will dramatically impact performance of our classifier, which is likely to labels white dogs as cats with _high degree of confidence_.

#### Dev/test set

If you suspect that there are many _incorrectly_ labeled examples in your dev or test set, you can add another column to your error analysis table where you track these incorrectly labeled examples. Depending on the total percentage of these examples, you can decide if it is worth the time to go through and correct all _incorrectly_ labeled examples in your dev or test set.

There are some special considerations when correcting incorrect dev/test set examples, namely:

- apply the same process to your dev and test sets to make sure they continue to come from the same distribution
- considering examining examples your algorithm got right as well as ones it got wrong
- train and dev/test data may now come from different distributions --- this is not necessarily a problem

#### Build quickly, then iterate

If you are working on a brand new ML system, it is recommended to _build quickly_, then _iterate_. For many problems, there are often tens or hundreds of directions we could reasonably choose to go in.


Building a system quickly breaks down to the following tasks:

1. set up a dev/test set and metric
2. build the initial system quickly and deploy
3. use bias/variance analysis & error analysis to prioritize next steps

A lot of value in this approach lies in the fact that we can quickly build insight to our problem.

> Note that this advice applies less when we have significant expertise in a given area and/or there is a significant body of academic work for the same or a very similar task (i.e., face recognition).

### Mismatched training and dev/test set

Deep learning algorithms are _extremely data hungry_. Because of this, some teams are tempted into shoving as much information into their training sets as possible. However, this poses a problem when the data sources do not come from the same distributions.

Lets illustrate this again with an example. Take our cat classifier. Say we have ~10,000 images from a **mobile app**, and these are the images (or _type_ of images) we hope to do well on. Assume as well that we have ~200,000 images from **webpages**, which have a slightly different underlying distribution than the mobile app images (say, for example, that they are generally higher quality.) _How do we combine these data sets?_

#### Option 1

We could take the all datasets, combine them, and shuffle them randomly into train/dev/test sets. However, this poses the obvious problem that _many of the examples in our dev set (~95% of them) will be from the webpage dataset_. We are effectively tuning our algorithm to a distribution that is _slightly different_ than our target distribution --- data from the mobile app.

#### Option 2

The second, recommended option, is to comprise the dev/test sets of images _entirely from the target (i.e., mobile data) distribution_. The advantage, is that we are now "aiming the target" in the right place, i.e., the distribution we hope to perform well on. The disadvantage of course, is that the training set comes from a different distribution than our target (dev/test) sets. However, this method is still superior to __option 1__, and we will discuss laters further ways of dealing with this difference in distributions.

> Note, we can still include examples from the distribution we care about in our training set, assuming we have enough data from this distribution.

### Bias and Variance with mismatched data distributions

Estimating the **bias** and **variance** of your learning algorithm can really help you prioritize what to work on next. The way you analyze bias and variance changes when your training set comes from a different distribution than your dev and test sets. Let's see how.

Let's keep using our cat classification example and let's say humans get near perfect performance on this. So, Bayes error, or Bayes optimal error, we know is nearly 0% on this problem. Assume further:

- training error: 1%
- dev error: 10%

If your **dev** data came from the _same distribution_ as your **training** set, you would say that you have a large **variance** problem, i.e., your algorithm is not generalizing well from the training set to the dev set. But in the setting where your training data and your dev data comes from a _different distribution_, you can no longer safely draw this conclusion. If the training and dev data come from _different underlying distributions_, then by comparing the training set to the dev set we are actually observing two different changes at the same time:

1. The algorithm _saw_ the training data. It did not _see_ the dev data
2. The data do not come from the same underlying distribution

In order to tease out these which of these is conributing to the drop in perfromsnce from our train to dev set, it will be useful to define a new piece of data which we'll call the **training-dev** set: a new subset of data with the same distribution as the training set, but not used for training.

Heres what we mean, previously we had train/dev/test sets. What we are going to do instead is randomly shuffle the training set and carve out a part of this shuffled set to be the **training-dev**.

![](https://s19.postimg.org/hytnawr6r/Screen_Shot_2018-02-25_at_12.10.35_PM.png)

> Just as the dev/test sets have the same distribution, the train-dev set and train set have the same distribution.

Now, say we have the following errors:

- training error: 1%
- train-dev error: 9%
- dev error: 10%

We see that training error \\(\lt \lt\\) train-dev error \\(\approx\\) dev error. Because the train and train-dev sets come from the same underlying distribution, we can safely conclude that the large increase in error from the train set to the dev set is due to _variance_ (i.e., our network is not generalizing well)

Lets look at a counter example. Say we have the following errors:

- training error: 1%
- train-dev error: 1.5%
- dev error: 10%

This is much more likely to be a _data mismatch problem_. Specifically, the algorithm is performing extremely well on the train and train-dev sets, but poorly on the dev set, hinting that the train/train-dev sets likely come from different underlying distributions than the dev set.

Finally, one last example. Say we have the following errors:

- Bayes error: \\(\approx\\) 0%
- training error: 10%
- train-dev error: 11%
- dev error: 20%

Here, we likely have two problems. First, we notice an _avoidable bias_ problem, suggested by the fact that our training error \\(\gt \gt\\) Bayes error. We also have a _data mismatch problem_, suggested by the fact that our training error \\(\approx\\) train-dev error by both are \\(\lt \lt\\) our dev error.

So let's take what we've done and write out the general principles. The _key quantities_ your want to look at are: human-level error (or Bayes error), training set error, training-dev set error and the dev set error.

The differences between these errors give us a sense about the **avoidable bias**, the **variance**, and the **data mismatch problem**. Generally,

- training error \\(\gt \gt\\) Bayes error: avoidable bias problem
- training error \\(\lt \lt\\) train-dev error: variance problem
- training error \\(\approx\\) train-dev error \\(\lt \lt\\) dev error: data mismatch problem.

#### More general formation

We can organize these metrics into a table; where the columns are different datasets (if you have more than one) and the rows are the error for examples the algorithm _was_ trained on and examples the algorithm _was not_ trained on.

![](https://s19.postimg.org/bfl8ak6xv/Screen_Shot_2018-02-25_at_4.34.00_PM.png)

### Addressing data mismatch

If your training set comes from a different distribution, than your dev and test set, and if error analysis shows you that you have a data mismatch problem, what can you do? Unfortunately, there are not (completely) systematic solutions to this, but let's look at some things you could try.

_Some recommendations:_

- carry out manual error analysis to try to understand different between training and dev/test sets.
  - _for example, you may find that many of the examples in your dev set are noisy when compared to those in your training set._
- make training data more similar; or collect more data similar to dev/test sets.
  - _for example, you may simulate noise in the training set_

The second point leads us into the idea of __artificial data synthesis__

#### Artificial data synthesis

In some cases, we may be able to artificially synthesis data to make up for a lack of real data. For example, we can imagine synthesizing images of cars to supplement a dataset of car images for the task of car recognition in photos.

[![artificial_car_images.png](https://s19.postimg.cc/dk5lbp60j/artificial_car_images.png)](https://postimg.cc/image/5rexjq01b/)

While artificial data synthesis can be a powerful technique for increasing the size of our dataset (and thus the performance of our learning algorithm), we must be wary of overfitting to the synthesized data. Say for example, the set of "all cars" and "synthesized cars" looked as follows:

[![artificial_data_venn.png](https://s19.postimg.cc/ojqsnbrar/artificial_data_venn.png)](https://postimg.cc/image/3mukint9r/)

In this case, we run a real risk of our algorithm overfitting to the synthesized images.

## Learning from multiple tasks

### Transfer learning

One of the most powerful ideas in deep learning is that you can take knowledge the neural network has learned from _one task_ and apply that knowledge to a _separate task_. So for example, maybe you could have the neural network learn to recognize objects like cats and then use parts of that knowledge to help you do a better job reading X-ray scans. This is called **transfer learning**. Let's take a look.

Lets say you have trained a neural network for __image recognition__. If you want to take this neural network and _transfer_ it to a different task, say radiology diagnosis, one method would be to _delete_ the last layer, and re-randomly initialize the weights feeding into the output layer.

To be concrete:

- during the first phase of training when you're training on an image recognition task, you train all of the usual parameters for the neural network, all the weights, all the layers
- having trained that neural network, what you now do to implement transfer learning is swap in a new data set \\(X,Y\\), where now these are radiology images and diagnoses pairs.
- finally, initialize the last layers' weights randomly and retrain the neural network on this new data set.

We have a couple options on how we retrain the dataset.

- If the radiology dataset is **small**: we should likely _"freeze"_ the transferred layers and only train the output layer.
- If the radiology dataset is __large__: we should likely train all layers.

> Sometimes, we call the process of training on the first dataset __pre-training__, and the process of training on the second dataset __fine-tuning__.

![](https://s19.postimg.org/ale9bpes3/Screen_Shot_2018-02-26_at_6.26.15_PM.png)

The idea is that learning from a very large image data set allows us to transfer some fundamental knowledge for the task of computer vision (i.e., extracting features such as lines/edges, small objects, etc.)

> Note that transfer learning is __not__ confined to computer vision examples, recent research has shown much success deploying transfer learning for NLP tasks.

#### When does transfer learning make sense?

Transfer learning makes sense when you have a _lot of data for the problem you're transferring **from** and usually relatively less data for the problem you're transferring **to**_.

So for our example, let's say you have a _million_ examples for image recognition task. Thats a lot of data to learn low level features or to learn a lot of useful features in the earlier layers in neural network. But for the radiology task, assume we only a hundred examples. So a lot of knowledge you learn from image recognition can be transferred and can really help you get going with radiology recognition even if you don't have enough data to perform well for the radiology diagnosis task.

If you're trying to learn from some **Task A** and transfer some of the knowledge to some **Task B**, then transfer learning makes sense when:

- Task A and B have the same input X.
- you have a lot more data for Task A than for Task B --- all this is under the assumption that what you really want to do well on is Task B.
- transfer learning will tend to make more sense if you suspect that low level features from Task A could be helpful for learning Task B.

### Multi-task learning

Whereas in transfer learning, you have a sequential process where you learn from task A and then transfer that to task B --- in multi-task learning, you start off simultaneously, trying to have one neural network do several things at the same time. The idea is that shared information from each of these tasks improves performance on _all_ tasks. Let's look at an example.

#### Simplified autonomous driving example

Let's say you're building an autonomous vehicle. Then your self driving car would need to detect several different things such as _pedestrians_, _other cars_, _stop signs_, _traffic lights_ etc.

Our input to the learning algorithm could be a single image, our our label for that example, \\(y^{(i)}\\) might be a four-dimensional column vector, where \\(0\\) at position \\(j\\) represents absence of that object from the image and \\(1\\) represents presence.

> E.g., a \\(0\\) at the first index of \\(y^{(i)}\\) might specify absence of a pedestrian in the image.

Our neural network architecture would then involve a single input and output layer. The twist is that the output layer would have \\(j\\) number of nodes, one per object we want to recognize.

![](https://s19.postimg.org/et91kny9v/Screen_Shot_2018-02-26_at_7.23.04_PM.png)

To account for this, our cost function will need to sum over the individual loss functions for each of the objects we wish to recongize:

\\[Cost = \frac{1}{m}\sum^m_{i=1}\sum^m_{j=1}\ell(\hat y_j^{(i)}, y_j^{(i)})\\]

> Were \\(\ell\\) is our logisitc loss.

Unlike traditional softmax regression, one image can have multiple labels. This, in essense, is __multi-task__ learning, as we are preforming multiple tasks with the same neural network (sets of weights/biases).

#### When does multi-task learning make sense?

Typically (but with some exceptions) when the following hold:

- Training on a set of tasks that could benefit from having shared lower-level features.
- Amount of data you have for each task is quite similar.
- Can train a big enough neural network to do well on all the tasks

> The last point is important. We typically need to "scale-up" the neural network in multi-task learning, as we will need a high variance model to be able to perform well on multiple tasks and typically more data --- as opposed to single tasks.

## End-to-end deep learning

One of the most exciting recent developments in deep learning has been the rise of **end-to-end deep** learning. So what is the end-to-end learning? Briefly, there have been some data processing systems, or learning systems that require _multiple stages of processing_. In contrast, end-to-end deep learning attempts to replace those multiple stages with a single neural network. Let's look at some examples.

### What is end-to-end deep learning?

#### Speech recognition example

At a high level, the task of speech recognition requires receiving as input some audio singles containing spoken words, and mapping that to a transcript containing those words.

Traditionally, speech recognition involved many stages of processing:

1. First, you would extract "hand-designed" features from the audio clip
2. Feed these features into a ML algorithm which would extract phonemes
3. Concatenate these phonemes to form words and then transcripts

In contrast to this step-by-step pipeline, __end-to-end deep learning__ seeks to model all these tasks with a single network given a set of inputs.

![](https://s19.postimg.org/cg5t1jlur/Screen_Shot_2018-02-27_at_8.53.09_PM.png)

The more traditional, **hand-crafted** approach tends to _outperform_ the **end-to-end approach** when _our dataset is small_, but this relationship flips as the dataset grows larger. Indeed, one of the biggest barriers to using end-to-end deep learning approaches is that large datasets which map our input to our final downstream task are _rare_.

> Think about this for a second and it makes perfect sense, its only recently in the era of deep learning that datasets have begun to map inputs to downstream outputs, skipping many of the intermediate levels of representation (images \\(\Rightarrow\\) labels, audio clips \\(\Rightarrow\\) transcripts.)

One example where end-to-end deep learning currently works very well is **machine translation** (massive, parallel corpuses have made end-to-end solutions feasible.)

### Summary

When end-to-end deep learning works, it can work really well and can simplify the system, removing the need to build many hand-designed individual components. But it's also not panacea, _it doesn't always work_.

### Whether or not to use end-to-end learning

Let's say in building a machine learning system you're trying to decide whether or not to use an end-to-end approach. Let's take a look at some of the pros and cons of end-to-end deep learning so that you can come away with some guidelines on whether or not an end-to-end approach seems promising for your application.

#### Pros and cons of end-to-end deep learning

__Pros__:

1. _let the data speak_: if you have enough labeled data, your network (given that it is large enough) should be able to a mapping from \\(x \rightarrow  y\\), with out having to rely on a humans preconceived notions or forcing the model to use some representation of the relationship between inputs an outputs.
2. _less hand-designing of components needed_: end-to-end deep learning seeks to model the entire task with a single learning algorithm, which typically involves little in the way of hand-designing components.

__Cons__:

1. _likely need a large amount of data for end-to-end learning to work well_
2. _excludes potentially useful hand-designed components_: if we have only a small training set, our learning algorithm likely does not have enough examples to learn representations that perform well. Although deep learning practitioners often speak despairingly about hand-crafted features or components, they allow us to inject priors into our model, which is particularly useful when we do not have a lot of labeled data.

> Note: hand-designed components and features are a double-edged sword. Poorly designed components may actually harm performance of the model by forcing the model to obey incorrect assumptions about the data.

#### Should I use end-to-end deep learning?

The key question we need to ask ourselves when considering on using end-to-end deep learning is:

_Do you have sufficient data to learn a function of the complexity needed to map \\(x\\) to \\(y\\)?_

Unfortunately, we do not have a formal definition of __complexity__ --- we have to rely on our intuition.

# Course 5: Sequence Models

#### TOC

1. [Week 1: Recurrent Neural Networks](#recurrent-neural-networks)
2. [Week 2: Introduction to Word Embeddings](#introduction-to-word-embeddings)

## Week 1: Recurrent Neural Networks

Learn about recurrent neural networks. This type of model has been proven to perform extremely well on temporal data. It has several variants including LSTMs, GRUs and Bidirectional RNNs, which you are going to learn about in this section.

### Why sequence models?

Recurrent neural networks (RNNs) have proven to be incredibly powerful networks for sequence modelling tasks (where the inputs \\(x\\), outputs \\(y\\) or both are sequences) including:

- speech recognition
- music generation
- sentiment classification
- DNA sequence analysis
- machine translation
- video activity recognition
- named entity recognition

#### Notation

As a motivating example, we will "build" a model that performs **named entity recognition** (**NER**).

_Example input_:

\\[x: \text{Harry Potter and Hermione Granger invented a new spell.} \\]

We want our model to output a target vector with the same number elements as our input sequence \\(x\\), representing the **named entities** in \\(x\\).

We will refer to each element in our input (\\(x)\\) and output (\\(y\\)) sequences with angled brackets, so for example, \\(x^{<1>}\\) would refer to "Harry". Because we have multiple input sequences, we denote the \\(i-th\\) sequence \\(x^{(i)}\\) (and its correpsonded output sequence \\(y^{(i)}\\)). The \\(t-th\\) element of the \\(i-th\\) input sequence is therefore \\(x^{(i)<t>}\\).

Let \\(T_x\\) be the length of the input sequence and \\(T_y\\) the length of the output sequence.

> In our example, \\(T_x\\) == \\(T_y\\)

#### Representing words

For NLP application, we have to decide on some way to represent words. Typically, we start by generating a __vocabulary__ (a dictionary of all the words that appear in our corpus).

> Note that in modern applications, a vocabulary of 30-50K is common and massive vocabularies (> 1 million word types) are often used in commercial applications, especially by big tech.

A common way to represent each word is to use a __one-hot__ encoding. In this way, we represent each token by a vector of dimension \\(||V||\\) (our vocabulary size).

_Example_:

\\[x^{<1>} = \begin{pmatrix} 0 \\\ ..\\\ 1 \\ ... \\\ 0 \end{pmatrix}\\]

\\(x^{<1>}\\) of our sequence (i.e. the token _Harry_) is represented as a vector which contains all zeros except for a single value of one at row \\(j\\), where \\(j\\) its position in the \\(V\\).

> "One-hot" refers to the fact that each vector contains only a single 1.

The goal is to learn a mapping from each \\(x^{<t>}\\) to some __tag__ (i.e. PERSON).

> Note that for out-of-vocabulary tokens, we typically assign a special value _\<UNK\>_ and a corresponding vector.

### Recurrent Neural Network Model

#### Why not a standard network?

In our previous example, we had 9 input words. You could imagine taking these 9 input words (represented are one-hot encoded vectors) as inputs to a "standard" neural network


![](https://s19.postimg.cc/x6zs848qr/Screen_Shot_2018-05-31_at_7.33.26_PM.png)

This turns out _not_ to work well. There are two main problems:

1. Inputs, outputs can be different __lengths__ in different examples (its not as if every \\(T_x, T_y\\) pair is of the same length).
2. A "standard" network doesn't **share features** learned across different positions of text. This is a problem for multiple reasons, but a big one is that this network architecture doesn't capture dependencies between elements in the sequence (e.g., the information that is a word in its context is not captured).

#### Recurrent Neural Networks

Unlike a "standard" neural network, **recurrent neural networks** (**RNN**) except input from the _previous_ timestep in a sequence. For our example \\(x\\) above, the _unrolled_ RNN diagram might look like the following:

![](https://s19.postimg.cc/zaa7gd103/Screen_Shot_2018-05-31_at_7.56.19_PM.png)

> Timestep 0 is usually initialized with a fake vector of 0's

Note that the diagram is sometimes drawn like this:

![](https://s19.postimg.cc/db3st5rvn/Screen_Shot_2018-05-31_at_7.50.01_PM.png)

Where the little black box represented a delay of _1 timestep_. Andrew prefers the unrolled diagrams (so do I!).

A RNN learns on a sequence from _left to right_, **sharing** the parameters from **each timestep**.

- the parameters governing the connection from \\(x^{<t>}\\) to the hidden layer will be some set of the parameters we're going to write as \\(W_{ax}\\).
- the activations, the horizontal connections, will be governed by some set of parameters \\(W_{aa}\\)
-  \\(W_{ya}\\), governs the output predictions

> Note: we take the notation \\(W_{ya}\\), to mean (for example) the parameters for variable \\(y\\) obtained by multiplying some quantity \\(a\\).

Notice, this parameter sharing means that when we make the prediction for, say \\(y^{<3>}\\), the RNN gets the information not only from \\(x^{<3>}\\) but also from the all the previous timesteps.

Note a potential weakness here. We don't incorporate information from previous timesteps in our predictions. This problem is solved by using **bidirectional** RNNs (BRNNs) which we discuss in a future video.

_Example_:

Given the sentences:

\\(x^{(1)}\\): _He said, "Teddy Roosevelt was a great President"_

\\(x^{(2)}\\): _He said, "Teddy bears are on sale!"_

And the task of __named entity recognition__ (__NER__), it would be really useful to know that the word "_President_" follows the name "_Teddy Roosevelt_" because as the second example suggest, using only _previous_ information in the sequence might not be enough to make a classification decision about an entity.

##### RNN Computation

Lets dig deeper into how a RNN works. First, lets start with a cleaned up depiction of our network

![](https://s19.postimg.cc/67vxdf4er/Screen_Shot_2018-06-01_at_10.06.12_AM.png)

__Forward Propagation__

Typically, we start off with the input \\(a^{<0>} = \vec 0\\). Then we perform our forward pass

- Compute our activation for timestep 1: \\(a^{<1>} = g(W_{aa}a^{<0>} + W_{ax}x^{<1>} + b_a)\\)
- Compute our prediction for timestep 1: \\(\hat y^{<1>} = g(W_{ya}a^{<1>} + b_y)\\)

More generally:
- \\(a^{<t>} = g(W_{aa}a^{<t-1>} + W_{ax}x^{<t>} + b_a)\\)
- \\(\hat y^{<t>} = g(W_{ya}a^{<t-1>} + b_y)\\)

> Where \\(b_a\\) is our bias vector.

The activation function used for the units of a RNN is most commonly __tanh__, although __ReLU__ is sometimes used. For the output units, it depends on our problem. Often __sigmoid__ / __softmax__ are used for binary and multi-class classification problems respectively.

#### Simplified RNN Notation

Lets take the general equations for forward propagation we developed above:

- \\(a^{<t>} = g(W_{aa}a^{<t-1>} + W_{ax}x^{<t>} + b_a)\\)
- \\(\hat y^{<t>} = g(W_{ya}a^{<t-1>} + b_y)\\)

1. We define our simplified **hidden activation** formulation:

\\[a^{<t>} = g(W_a[a^{<t-1>}, x^{<t>}] + b_a)\\]

Where
- \\(W_a = \begin{pmatrix} W_{aa} & | & W_{ax} \end{pmatrix}\\)
- \\([a^{<t-1>}, x^{<t>}] = \begin{pmatrix} a^{<t-1>} \\\ x^{<t>} \end{pmatrix}\\)

> Note that \\(W_a[a^{<t-1>}, x^{<t>}] = W_{aa}a^{<t-1>} + W_{ax}x^{<t>}\\)

The advantages of this notation is that we can compress two parameter matrices into one.

2. And our simplified __output activation__ formulation:

\\[\hat y^{<t>} = g(W_{y}a^{<t-1>} + b_y)\\]

### Backpropagation through time

#### Forward propagation

We have seen at a high-level how forward propagation works for an RNN. Essentially, we forward propagate the input, multiplying it by our weight matrices and applying our activation for each timestep until we have outputted a prediction for each timestep in the input sequence.

More explicitly, we can represent the process of foward propogation as a series of matrix multiplications in diagram form:

![](https://s19.postimg.cc/u058ol3r7/Screen_Shot_2018-06-01_at_1.06.39_PM.png)

#### Backward propogation through time (BPTT)

In order to perform **backward propagation through time** (**BPTT**), we first have to specify a loss function. We will choose __cross-entropy__ loss (we also saw this when discussing logisitc regression):

_For a single prediction (timestep)_

\\[\ell^{<t>}(\hat y^{<t>}, y^{<t>}) = -y^{<t>} \log \hat y^{<t>} - (1 - y^{<t>}) \log(1- \hat y^{<t>})\\]

_For all predictions_

\\[\ell = \sum_{t=1}^{T_y}\ell^{<t>}(\hat y^{<t>}, y^{<t>})\\]

While not covered in detail here, BPTT simply involves applying our loss function to each prediction at each timestep, and then using this information along with the chain rule to compute the gradients we will need to update our parameters and _assign blame proportionally_. The entire process might look something like the following:

![](https://s19.postimg.cc/44li5lc9v/Screen_Shot_2018-06-01_at_1.17.38_PM.png)

### Different types of RNNs

So far, we have only seen an RNN where the input and output are both sequences of lengths > 1. Particularly, our input and output sequences were of the same length (\\(T_x == T_y\\)), For many applications, \\(T_x \not = T_y\\).

Take **sentiment classification** for example, where the input is typically a sequence (of text) and the output a integer scale (a 1-5 star review, for example).

_Example_:

\\(x\\): "There is nothing to like in this movie"

We want our network to output a single prediction from 1-5. This is an example of a __many-to-one__ architecture.

![](https://s19.postimg.cc/qukmyfy0j/Screen_Shot_2018-06-01_at_1.32.36_PM.png)

Another type of RNN architecture is __one-to-many__. An example of this architecture is __music generation__, where we might input an integer (indicating a genre) or the 0-vector (no input) and generate musical notes as our output. In this case, we input a single value to the network at timestep 1, and then propagate that input through the network (the remaining timesteps), with the caveat that in this architecture, we often take the ouput from the previous timestep and feed it to the next timestep:

![](https://s19.postimg.cc/4imu5141f/Screen_Shot_2018-06-01_at_1.39.52_PM.png)

The final example is a __many-to-many__ architecture. Unlike our previous example where \\(T_x == T_y\\), in machine translation \\(T_x \not = T_y\\), as the number of words in the input sentence (say, in _english_) is not necessarily the same as the output sentence (say, in _french_). These problems are typicaly solved with __sequence to sequence models__, that are composed of distinct __encoder__ and __decoder__ RNNs.

![](https://s19.postimg.cc/mlfwwes83/Screen_Shot_2018-06-01_at_1.47.32_PM.png)

#### Summary of RNN types

1. **One-to-one**: a standard, generic neural network. Strictly speaking, you wouldn't model this problem with an RNN.

![](https://s19.postimg.cc/g7qtta5df/Screen_Shot_2018-06-01_at_1.49.36_PM.png)

2. __One-to-many__: Where our input is a single value (or in some cases, a null input represented by the 0-vector) which propogates through the network and our output is a sequence. Often, we use the prediction from the previous timestep when computing the hidden activations. An example is _music generation_ or _sequence generation_ more generally.

![](https://s19.postimg.cc/efxuyd6kz/Screen_Shot_2018-06-01_at_1.51.52_PM.png)

3. __Many-to-one__: Where our input is a sequence and our output is a single value. Typically we take the prediction from the last timestep of the RNN. An example is _sentiment classification_

![](https://s19.postimg.cc/9u1qq0dc3/Screen_Shot_2018-06-01_at_1.53.11_PM.png)

4. __Many-to-many__: Where both our input and outputs are sequences. These sequence are not necciscairly the same length (\\(T_x \not = T_y\\)).

  - When \\(T_x == T_y\\) our architecture looks like a standard RNN:

![](https://s19.postimg.cc/5xoetzxhf/Screen_Shot_2018-06-01_at_1.55.17_PM.png)

  - and when \\(T_x \not = T_y\\) are architecture is a _sequence to sequence_ model which looks like:

![](https://s19.postimg.cc/585mhn4nn/Screen_Shot_2018-06-01_at_1.55.13_PM.png)

### Language model and sequence generation

**Language modeling** is one of the most basic and important tasks in natural language processing. It's also one that RNNs handle very well.

#### What is a language modeling?

Let's say you are building a __speech recognition__ system and you hear the sentence:

\\[\text{"The apple and pear/pair salad"}\\]

How does a neural network determine whether the speaker said _pear_ or _pair_ (never mind that the correct answer is obvious to us). The answer is that the network encodes a __language model__. This language model is able to determine the _probability_ of a given sentence (think of this as a measure of "correctness" or "goodness"). For example, our language model might output:

\\[P(\text{The apple and pair salad}) = 3.2 \times 10^{-13}\\]
\\[P(\text{The apple and pear salad}) = 5.7 \times 10^{-10}\\]

This system would then pick the much more likely second option.

#### Language modeling with an RNN

We start with a large corpus of english text. The first step is to __tokenize__ the text in order to form a vocabulary

\\[\text{"Cats average 15 hours of sleep a day"} \rightarrow \text{["Cats", "average", "15", "hours", "of", "sleep", "a", "day", "."]}\\]

These tokens are then __one-hot encoded__ or mapped to __indices__. Sometimes, a special end-of-sentence token is appended to each sequence (\<EOS\>).

> What if some of the words we encounter are not in our vocabulary? Typically we add a special token, \<UNK\> to deal with this problem.

Finally, we build an RNN to model the likelihood of any given sentence, learned from the training corpus.

##### RNN model

At time 0, we compute some activation \\(a^{<1>}\\) as a function of some inputs \\(x^{<1>}\\). In this case, \\(x^{<1>}\\) will just be set to the zero vector. Similarly, \\(a^{<0>}\\), by convention, is also set to the zero vector.

\\(a^{<1>}\\) will make a _softmax_ prediction over the entire vocabulary to determine \\(\hat y^{<1>}\\) (the probability of observing any of the tokens in your vocabulary as the _first_ word in a sentence).

At the second timestep, we will actually feed the first token in the sequence as the input (\\(x^{<2>} = y^{<1>}\\)). This occurs, so forth and so on, such that the input to each timestep are the tokens for all previous timesteps. Our outputs \\(\hat y^{<t>}\\) are therefore  \\(P(x^{<t>}|x^{<t-1>}, x^{<t-2>}, ..., x^{<t-n>})\\) where \\(n\\) is the length of the sequence.

> Just a note here, we are choosing \\(x^{<t>} = y^{<t-1>}\\) _NOT_ \\(x^{<t>} = \hat y^{<t-1>}\\)

The full model looks something like:

[![Screen_Shot_2018-06-01_at_6.40.05_PM.png](https://s19.postimg.cc/5c8mpbc0z/Screen_Shot_2018-06-01_at_6.40.05_PM.png)](https://postimg.cc/image/sqgm18ty7/)

There are two important steps in this process:

1) Estimate \\(\hat y^{<t>} = P(y^{<t>} | y^{<1>}, y^{<2>}, ..., y^{<t-1>})\\)
2) Then pass the ground-truth word from the training set to the next time-step.

The __loss function__ is simply the __cross-entropy__ lost function that we saw earlier:

- For single examples: \\(\ell(\hat y^{<t>}, y^{<t>}) = - \sum_i y_i^{<t>} \log \hat y_i^{<t>}\\)
- For the entire training set: \\(\ell = \sum_i \ell^{<t>}(\hat y^{<t>}, y^{<t>})\\)

Once trained, the RNN will be able to predict the probability of any given sentence (we simply multiply the probabilities output by the RNN at each timestep).

### Sampling novel sequences

After you train a sequence model, one of the ways you can get an informal sense of what is learned is to sample novel sequences (also known as an _intrinsic evaluation_). Let's take a look at how you could do that.

#### Word-level models

Remember that a sequence model models the probability of any given sequence of words. What we would like to to is to _sample_ from this distribution to generate _novel_ sequence of words.

> At this point, Andrew makes a distinction between the architecture used for _training_ a language modeling and the architecture used for _sampling_ from a language model. The distinction is completely lost on me.

We start by computing the activation \\(a^{<1>}\\) as a function of some inputs \\(x^{<1>}\\) and \\(a^{<0>}\\) (again, these are set to the zero vector by convention). The __softmax__ function is used to generate a probability distribution over all words in the vocabulary, representing the likelihood of seeing each at the first position of a word sequence. We then randomly sample from this distribution, choosing a single token (\\(\hat y^{<1>}\\)), and pass it as input for the next timestep.

> For example, if we sampled "the" in the first timestep, we would set \\(\hat y^{<1>} = the = x^{<2>}\\). This means that at the second timestep, we are computing a probability distribution \\(P(v | the)\\) over all tokens \\(v\\) in our vocabulary \\(V\\).

The entire procedure looks something like:

[![Screen_Shot_2018-06-02_at_12.31.59_PM.png](https://s19.postimg.cc/c2gdqplyb/Screen_Shot_2018-06-02_at_12.31.59_PM.png)](https://postimg.cc/image/cf7rww47z/)

_How do we know when the sequence ends_?

If we included the \<EOS\> token in our training procedure (and this included it in our vocabulary) the sequence ends when and \<EOS\> token is generated. Otherwise, stop when a pre-determined number of tokens has been reached.

_What if we generate an \<UNK\> token_?

We can simply re-sample until we generate a non-\<UNK\> token.

#### Character-level models

We could also build a **character-level language model**. The only major difference is that we train on a sequence of _characters_ as opposed to _tokens_, and therefore our vocabulary consists of individual _characters_ (which typically include digits, punctuation, etc.)

_Character_-level language models are more computational expensive, and because a sequence of characters is typically much longer than a sequence of words (obviously) it is more difficult to capture the long range dependencies (as they are longer, of course).

However, using a _character_-level language models has the benefit of avoiding the problem of out-of-vocabulary tokens, as we can build a non-zero vector representation of _any_ token using the learned character representations.

> Note, you can also combine word-level and character-level language models!

### Vanishing gradients with RNNs

One of the problems with the basic RNN algorithm is the __vanishing gradient problem__. The RNN architecture as we have described it so far:

[![Screen_Shot_2018-06-02_at_12.57.39_PM.png](https://s19.postimg.cc/xddxv1wdv/Screen_Shot_2018-06-02_at_12.57.39_PM.png)](https://postimg.cc/image/th0lz2be7/)

Take the following two input examples:

\\[x^{(1)} = \text{The cat, which already ate ..., was full}\\]
\\[x^{(2)} = \text{The cats, which already ate ..., were full}\\]

> Take the "..." to be an sequence of english words of arbitrary length.

Cleary, there is a long range-dependency here between the _grammatical number_ of the **noun** "cat" and the _grammatical number_ of the **verb** "was".

> It is important to note that while this is a contrived example, language very often contains long-range dependencies.

It turns out that the basic RNN that we have described thus far is not good at capturing such long-range dependencies. To explain why, think back to our earlier discussions about the _vanishing gradient problems_ in very deep neural networks. The basic idea is that in a network with many layers, the gradient becomes increasingly smaller as it is backpropagated through a very deep network, effectively "vanishing". RNNs face the same problem, leading to errors in the outputs of later timesteps having little effect on the gradients of earlier timesteps. This leads to a failure to capture long-range dependencies.

> Because of this problem, a basic RNN captures mainly local influences.

Recall that _exploding gradients_ are a similarly yet opposite problem. It turns out that _vanishing gradients_ are a bigger problems for RNNs, but _exploding gradients_ do occur. However,  _exploding gradients_ are typically easier to catch as we simply need to look for gradients that become very very large (also, they usually lead to computational overflow, and generate NaNs). The solution to _exploding gradient_ problems is fairly straightforward however, as we can use a technique like __gradient clipping__ to scale our gradients according to some maximum values.

### Gated Recurrent Unit (GRU)

You've seen how a basic RNN works. In this video, you learn about the **Gated Recurrent Unit** (**GRU**) which is a modification to the RNN hidden layer that makes it much better capturing long range connections and helps a lot with the vanishing gradient problems.

Recall the activate function for an RNN at timestep \\(t\\):

\\[a^{<t>} = g(W_a[a^{<t-1>}, x^{<t>}] + b_a)\\]

As a picture:

[![rnn_unit_hand_drawn.png](https://s19.postimg.cc/5nwil5l9v/rnn_unit_hand_drawn.png)](https://postimg.cc/image/ceczul8fj/)

> Two papers were important for the development of GRUs: [Cho et al., 2014](https://arxiv.org/pdf/1406.1078v3.pdf) and [Chung et al., 2014](https://arxiv.org/pdf/1412.3555.pdf).

Lets define a new variable, \\(c\\) for the **memory cell**. The job of the memory cell is to remember information earlier in a sequence. So at time \\(<t>\\) the memory cell will have some value \\(c^{<t>}\\). In GRUs, it turns out that \\(c^{<t>} == a^{<t>}\\).

> It will be useful to use the distinct variables however, as in LSTM networks \\(c^{<t>} \not = a^{<t>}\\)

At every timestep \\(t\\), we are going to consider overwriting the value of the memory cell \\(c^{<t>}\\) with a new value, computed with an activation function:

\\[\text{Candidate memory: }\tilde c^{<t>} = tanh(W_c[c^{<t-1>}, x^{<t>}] + b_c)\\]

The most important idea in the GRU is that of an __update gate__, \\(\Gamma_u\\), which always has a value between 0 and 1:

\\[\Gamma_u = \sigma(W_u[c^{<t-1>}, x^{<t>}] + b_u)\\]

> Subscript \\(u\\) stands for update.

To build our intuition, think about the example we introduced earlier:

\\[x^{(1)} = \text{The cat, which already ate ..., was full}\\]

We noted that here, the fact that the word _"cat"_ was singular was a huge hint that the verb _"was"_ would also be singular in number. We can imagine \\(c^{<t>}\\) as _memorizing_ the case of the noun "_cat_" until it reached the verb "_was_". The job of the __gate__ would be to _remember_ this information between "_... cat ... were ..._" and _forget_ it afterwords.

To compute \\(c^{<t>}\\):

\\[c^{<t>} = \Gamma_u * \tilde c^{<t>} + (1 - \Gamma_u) * c^{<t>}\\]

There is a very intuitive understanding of this computation. When \\(\Gamma_u\\) is 1, we simply _forget_ the old value of \\(c^{<t>}\\) by overwriting it with \\(\tilde c^{<t>}\\). When \\(\Gamma_u\\) is 0, we do the opposite (completely diregard the new candidate memory \\(\tilde c^{<t>}\\) in favour of the old memory cell value \\(c^{<t>}\\)).

> Remember that \\(\Gamma_u\\) can take on any value between 0 and 1. The larger the value, the more weight that the candidate memory cell value takes over the old memory cell value.

For our example sentence above, we might hope that the GRU would set \\(\Gamma_u = 1\\) once it reached "_cats_", and then \\(\Gamma_u = 0\\) for every other timestep until it reached "_was_", where it might set \\(\Gamma_u = 1\\) again. Think of this as the network memorizing the grammatical number of the **subject** of the sentence in order to determine the number of its verb, a concept known as __agreement__.

As a picture:

[![gru_unit_hand_drawn.png](https://s19.postimg.cc/3xi2xfg9v/gru_unit_hand_drawn.png)](https://postimg.cc/image/v83e5cj6n/)

> The purple box just represents our calculation of \\(c^{<t>}\\)

GRUs are remarkably good at determining when to update the memory cell in order to **memorize** or **forget** information in the sequence.

#### Vanishing gradient problem

The way a GRU solves the vanishing gradient problem is straightforward: the __memory cell__ \\(c^{<t>}\\) is able to retain information over many timesteps. Even if \\(\Gamma_u\\) becomes very very small, \\(c^{<t>}\\) will essentially retain its value across many many timesteps.

#### Implementation details

\\(c^{<t>}, \tilde c^{<t>} \text{ and } \Gamma_u\\) are all vectors of the same dimension. This means that in the computation of:

\\[c^{<t>} = \Gamma_u \ast \tilde c^{<t>} + (1 - \Gamma_u) \ast c^{<t>}\\]

\\(\ast\\) are _element-wise_ multiplications. Thus, if \\(\Gamma_u\\) is a 100-dimensional vector, it is really a 100-dimensional vector of _bits_ which tells us of the 100-dimensional memory cell \\(c^{<t>}\\), which are the _bits_ we want to _update_.

> Of course, in practice \\(\Gamma_u\\) will take on values that are not _exactly_ 0 or 1, but its helpful to image it as a bit vector to build our intuition.

Invoking our earlier example one more time:

\\[x^{(1)} = \text{The cat, which already ate ..., was full}\\]

we could imagine representing the grammatical number of the noun "_cat_" as a single _bit_ in the memory cell.

#### Full GRU unit

The description of a GRU unit provided above is actually somewhat simplified. Below is the computations for the _full_ GRU unit:

\\[\Gamma_r = \sigma(W_r[c^{<t-1>}, x^{<t>}] + b_r)\\]
\\[\tilde c^{<t>} = tanh(W_c[\Gamma_r \ast c^{<t-1>}, x^{<t>}] + b_c)\\]
\\[\Gamma_u = \sigma(W_u[c^{<t-1>}, x^{<t>}] + b_u)\\]
\\[c^{<t>} = \Gamma_u * \tilde c^{<t>} + (1 - \Gamma_u) * c^{<t-1>}\\]

We introduce another gate, \\(\Gamma_r\\). Where we can think of this gate as capturing how relevant \\(c^{<t-1>}\\) is for computing the next candidate \\(c^{<t>}\\).

> You can think of \\(r\\) as standing for relevance.

Note that Andrew tried to establish a consistent notation to use for explaining both GRUs and LSTMs. In the academic literature, you might often see:

- \\(\tilde c^{<t>} : \tilde h\\)
- \\(\Gamma_u : u\\)
- \\(\Gamma_u : r\\)
- \\(c^{<t>} : h\\)

> (our notation : common academic notation)

### Long Short Term Memory (LSTM)

In the last video, you learned about the **GRU**, and how that can allow you to learn very long range dependencies in a sequence. The other type of unit that allows you to do this very well is the **LSTM** or the **long short term memory** units, and is even more powerful than the GRU.

Recall the full set of equations defining a GRU above:

\\[\Gamma_r = \sigma(W_r[c^{<t-1>}, x^{<t>}] + b_r)\\]
\\[\tilde c^{<t>} = tanh(W_c[\Gamma_r \ast c^{<t-1>}, x^{<t>}] + b_c)\\]
\\[\Gamma_u = \sigma(W_u[c^{<t-1>}, x^{<t>}] + b_u)\\]
\\[c^{<t>} = \Gamma_u * \tilde c^{<t>} + (1 - \Gamma_u) * c^{<t-1>}\\]
\\[a^{<t>} = c^{<t>}\\]

The LSTM unit is a more powerful and slightly more general version of the GRU (in truth, the LSTM was defined before the GRU). Its computations are defined as follows:

\\[\tilde c^{<t>} = tanh(W_c[a^{<t-1>}, x^{<t>}] + b_c)\\]
\\[\Gamma_u = \sigma(W_u[a^{<t-1>}, x^{<t>}] + b_u)\\]
\\[\Gamma_f = \sigma(W_f[a^{<t-1>}, x^{<t>}] + b_f)\\]
\\[\Gamma_o = \sigma(W_o[a^{<t-1>}, x^{<t>}] + b_o)\\]
\\[c^{<t>} = \Gamma_u * \tilde c^{<t>} + \Gamma_f * c^{<t-1>}\\]
\\[a^{<t>} = \Gamma_o  * tanh(c^{<t>})\\]

> [Original LSTM paper](https://dl.acm.org/citation.cfm?id=1246450).

Notice that with LSTMs, \\(a^{<t>} \not = c^{<t>}\\). One new property of the LSTM is that instead of _one_ update gate, \\(\Gamma_u\\), we have _two_ update gates, \\(\Gamma_u\\) and \\(\Gamma_f\\) (for **update** and **forget** respectively). This gives the memory cell the option of keeping the old memory cell information \\(c^{<t-1>}\\) and just adding to it some new information \\(\tilde c^{<t>}\\).

We can represent the LSTM unit in diagram form as follows:

[![lstm_unit_clean.png](https://s19.postimg.cc/6bfomuc9v/lstm_unit_clean.png)](https://postimg.cc/image/f6gixd127/)

> See [here](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) for an more detailed explanation of an LSTM unit.

One thing you may notice is that if we draw out multiple units in temporal succession, it becomes clear how the LSTM is able to achieve something akin to "memory" over a sequence:

[![multiple_lstm_units_clean.png](https://s19.postimg.cc/iq2gn6bhv/multiple_lstm_units_clean.png)](https://postimg.cc/image/mz76pcer3/)

#### Modifications to LSTMs

There are many modifications to the LSTM described above. One involves including \\(c^{<t-1>}\\) along with \\(a^{<t-1>}, x^{<t>}\\) in the gate computations, known as a __peephole connection__. This allows for the gate values to depend not just on the input and the previous timesteps activation, but also on the previous timesteps value of the memory cell.

### Bidirectional RNNs (BRNNs)

By now, you've seen most of the building blocks of RNNs. There are two more ideas that let you build much more powerful models. One is **bidirectional RNNs** (**BRNNs**), which lets you at a point in time to take information from both earlier and later in the sequence. And second, is deep RNNs, which you'll see in the next video.

To motivate bidirectional RNNs, we will look at an example we saw previously:

\\(x^{(1)}\\): _He said, "Teddy Roosevelt was a great President"_

\\(x^{(2)}\\): _He said, "Teddy bears are on sale!"_

Recall, that for the task of NER we established that correctly predicting the token _Teddy_ as a _person_ entity without seeing the words that follow it would be difficult.

[![rnn_teddy_example.png](https://s19.postimg.cc/i4i34ql83/rnn_teddy_example.png)](https://postimg.cc/image/m0vf0q67j/)

> Note: this problem is independent of whether these are standard RNN, GRU, or LSTM units.

A solution to this problem is to introduce another RNN in the opposite direction, going _backwards_ in time.

[![bidirectional_rnn.png](https://s19.postimg.cc/99h8u7opv/bidirectional_rnn.png)](https://postimg.cc/image/o5fs1t04f/)

During forward propogation, we compute activations as we have seen previously, with key difference being that we learn two series of activations: one from _left-to-right_ \\(\overrightarrow a^{<t>}\\) and one from _right-to-left_ \\(\overleftarrow a^{<t>}\\). What this allows us to do is learn the representation of each element in the sequence _within its context_. Explicitly, this is done by using the output of both the forward and backward units at each time step in order to make a prediction \\(\hat y^{<t>}\\):

\\[\hat y^{<t>} = g(W_y[\overrightarrow a^{<t>}, \overleftarrow a^{<t>}] + b_y)\\]

For the example given above, this means that our prediction for the token _Teddy_, \\(y^{<3>}\\), is able to makes use of information seen previously in the sequence (\\(t = 3, 2, ...\\)) and future information in the sequence (\\(t = 4, 5, ...\\))

> Note again that we can build bidirectional networks with standard RNN, GRU and LSTM units. Bidirectional LSTMs are extremely common.

The disadvantage of BRNNs is that we need to see the _entire_ sequence before we can make any predictions. This can be a problem in applications such as real-time speech recognition.

> the BRNN will let you take into account the entire speech utterance but if you use our straightforward implementation, you need to wait for the person to stop talking to get the entire utterance before you can actually process it and make a speech recognition prediction

For applications like these, there exists somewhat more complex modules that allow predictions to be made before the full sequence has been seen. bidirectional RNN as you've seen here. For many NLP applications where you can get the entire sentence all the same time, our standard BRNN algorithm is actually very effective.

### Deep RNNs

The different versions of RNNs you've seen so far will already work quite well by themselves. But for learning very complex functions sometimes is useful to stack multiple layers of RNNs together to build even deeper versions of these models.

Recall, that for a standard neural network we have some input \\(x\\) which is fed to a hidden layer with activations \\(a^{[l]}\\) which are in turn fed to the next layer to produce activations \\(a^{[l+1]}\\). In this was, we can stack as many layers as we like. The same is true of RNNs. Lets use the notation \\(a^{[l]<t>}\\) to denote the activations of layer \\(l\\) for timestep \\(t\\).

A stacked RNN would thus look something like the following:

[![deep_rnn_clean.png](https://s19.postimg.cc/qck0pmpyb/deep_rnn_clean.png)](https://postimg.cc/image/mg6otn4yn/)

The computation of, for example, \\(a^{[2]<3>}\\) would be:

\\[a^{[2]<3>} = g(W_a^{[2]}[a^{[2]<2>}, a^{[1]<3>}] + b_a^{[2]})\\]

Notice that the second layer has parameters \\(W_a^{[2]}\\) and \\(b_a^{[2]}\\) which are shared across all timesteps, but _not_ across the layers (which have their own corresponding set of parameters).

Unlike standard neural networks, we rarely stack RNNs very deep. Part of the reason is that RNNs are already quite large due to their temporal dimension.

> A common depth would be 2 stacked RNNs.

Something that has become more common is to apply deep neural networks to the output of each timestep. In this approach, the _same_ deep neural network is typically applied to each output of the final RNN layer.

## Week 2: Natural Language Processing & Word Embeddings

Natural language processing and deep learning is an _important combination_. Using word vector representations and embedding layers, you can train recurrent neural networks with outstanding performances in a wide variety of industries. Examples of applications are **sentiment analysis**, **named entity recognition** (**NER**) and **machine translation**.

### Introduction to word embeddings: Word Representation

Last week, we learned about RNNs, GRUs, and LSTMs. In this week, you see how many of these ideas can be applied to **Natural Language Processing** (**NLP**), which is one of the areas of AI being revolutionized by deep learning. One of the key ideas you learn about is **word embeddings**, which is a way of representing words.

So far, we have been representing words with a vocabulary, \\(V\\), of one-hot-encoded vectors. Lets quickly introduce a new notation. If the token "_Man_" is in position 5391 in our vocabulary \\(V\\) then we denote the corresponding one-hot-encoded vector as \\(O_{5391}\\).

One of the weaknesses of this representation is that it treats each word as a "thing" onto itself, and doesn't allow a language model to generalize between words. Take the following examples:

\\[x_1: \text{"I want a glass of orange juice"}\\]
\\[x_2: \text{"I want a glass of apple juice"}\\]

Cleary, the example sentences are extremely semantically similar. However, in a one-hot encoding scheme, a model which has learned that \\(x_1\\) is a likely sentence is unable to fully generalize to example \\(x_2\\), as the relationship between "_apple_" and "_orange_" is not any closer than the relationship between "_orange_" and any other word in the vocabulary.

Notice, in fact, that the [inner product](http://www.wikiwand.com/en/Dot_product) between any two one-hot encoded vectors:

\\[O_i \times O_j = \vec 0 \text{ for } \forall i,j\\]

And similarly, the [euclidean distance](http://www.wikiwand.com/en/Euclidean_distance) between any two one-hot encoded vectors is identical:

\\[||O_i - O_j|| = \sqrt{|V|}\text{ for } \forall i,j\\]

To build our intuition of word embeddings, image a contrived example where we represent each word with some __feature representation__:

[![word_embeddings_intro.png](https://s19.postimg.cc/6bvaxkjw3/word_embeddings_intro.png)](https://postimg.cc/image/5mcil7jcf/)

We could imagine many features (with values -1 to 1, say) that can be used to build up a featue representation, an \\(f_n\\)-dimensional vector, of each word. Similarly to our one-hot representations, lets introduce a new notation \\(e_i\\) to represent the _embedding_ of token \\(i\\) in our vocabulary \\(V\\).

> Where \\(f_n\\) is the number of features.

Thinking back to our previous example, notice that our representations for the tokens "_apple_" and "_orange_" become quite similar. This is the critical point, and what allows our language model to generalize between word tokens and even entire sentences.

> In the later videos, we will see how to learn these embeddings. Note that the learned representations do not have an easy interpretation like the dummy embeddings we presented above.

#### Visualizing word embeddings

Once these feature vectors or _embeddings_ are learned, a popular thing to do is to use dimensionality reduction to _embed_ them into a 2D geometric space for easy visualization. An example of this using our word representations presented above:

[![visualize_embeddings.png](https://s19.postimg.cc/e5vwinmgj/visualize_embeddings.png)](https://postimg.cc/image/ofybhwcbz/)

We notice that semantically similar words tend to cluster together, and that each cluster seems to roughly represent some idea or concept (i.e., numbers typically cluster together). This demonstrates our ability to learn _similar_ feature vectors for _similar_ tokens and will allow our models to generalize between words and even sentences.

> A common algorithm for doing this is the [t-SNE](http://www.wikiwand.com/en/T-distributed_stochastic_neighbor_embedding) algorithm.

The reason this feature representations are called _embeddings_ is because we imagine that we are _embedding_ each word into a geometric space (say, of 300 dimensions). If you imagine a cube, we can think of giving each word a single unit of space within this cube.

[![why_embeddings.png](https://s19.postimg.cc/ofybi1237/why_embeddings.png)](https://postimg.cc/image/llv64kzwv/)

### Introduction to word embeddings: Using word embeddings

In the last lecture, you saw what it might mean to learn a featurized representations of different words. In this lecture, you see how we can take these representations and plug them into NLP applications.

#### Named entity recognition example

Take again the example of named entity recognition, and image we have the following example:

[![ner_word_emb_example.png](https://s19.postimg.cc/btx3cxm6b/ner_word_emb_example.png)](https://postimg.cc/image/9clc5o29r/)

Let's assume we correctly identify "_Sally Johnson_" as a PERSON entity. Now imagine we see the following sequence:

\\[x: \text{"Robert Lin is a durian cultivator"}\\]

> Note that durian is a type of fruit.

In all likelihood, a model using word embeddings as input should be able to generalize between the two input examples, a take advantage of the fact that it previously labeled the first two tokens of a similar training example ("_Sally Johnson_") as a PERSON entity. But how does the model generalize between "_orange farmer_" and "_durian cultivator_"?

Because word embeddings are typically trained on _massive_ unlabeled text corpora, on the scale of 1 - 100 billion words. Thus, it is likely that the word embeddings would have seen and learned the similarity between word pairs ("_orange_", "_durian_") and ("_farmer_", "_cultivator_").

In truth, this method of __transfer learning__ is typically how we use word embeddings in NLP tasks.

#### Transfer learning and word embeddings

How exactly do we utilize transfer learning of word embeddings for NLP tasks?

1. Learn word embeddings from large text corpus (1-100B words), OR, download pre-trained embeddings online.
2. Transfer the embedding to a new task with a (much) smaller training set (say, 100K words).
3. (Optional): Continue to fine-tune the word embeddings with new data. In practice, this is only advisable if your training dataset is quite large.

This method of transfer learning with word embeddings has found use in NER, text summarization, co-reference resolution, and parsing. However, it has been less useful for language modeling and machine translation (especially when a lot of data for these tasks is available).

One major advantage to using word embeddings to represent tokens is that it reduces the dimensionality of our inputs, compared to the one-hot encoding scheme. For example, a typical vocabulary may be 10,000 or more word types, while a typical word embedding may be around 300 dimensions.

### Introduction to word embeddings: Properties of word embeddings

By now, you should have a sense of how word embeddings can help you build NLP applications. One of the most fascinating properties of word embeddings is that they can also help with **analogy reasoning**. And while reasoning by analogy may not be, by itself, the most important NLP application, it helps to convey a sense of what information these word embeddings are capturing.

#### Analogies

Let us return to our previous example:

[![word_emb_analogies.png](https://s19.postimg.cc/b6y6n9cdv/word_emb_analogies.png)](https://postimg.cc/image/wglsy3sof/)

Say we post the question: "_Man is to women as king is to **what**?_"

Many of us would agree that the answer to this question is "Queen" (in part because of humans remarkable ability to [reason by analogy](https://plato.stanford.edu/entries/reasoning-analogy/)). But can we have a computer arrive at the same answer using embeddings?

First, lets simplify our earlier notation and allow \\(e_{man}\\) to denote the learned embedding for the token "_man_". Now, if we take the difference \\(e_{man} - e_{woman}\\), the resulting vector is closest to \\(e_{king} - e_{queen}\\).

> Note you can confirm this using our made up embeddings in the table.

Explicitly, an algorithm to answer the question "_Man is to women as king is to **what**?_" would involve computing \\(e_{man} - e_{woman}\\), and then finding the token \\(w\\) that produces \\(e_{man} - e_{woman} \approx e_{king} - e_{w}\\).

> This ability to mimic analogical reasoning and other interesting properties of word embeddings were introduced in this [paper](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/rvecs.pdf).

Lets try to visualize why this makes sense. Imagine our word embedding plotted as vectors in a 300D space (represented here in 2 dimensions for visualization). We would expect our vectors to line up in a parallelogram:

[![visualize_word_emb_300d.png](https://s19.postimg.cc/f8f9t59w3/visualize_word_emb_300d.png)](https://postimg.cc/image/c1kq9ipfz/)


Note that in reality, if you use a dimensionality reduction algorithm such as t-SNE, you will find that this expected relationship between words in an analogy does not hold:

[![t-sne_visulize_word_emb.png](https://s19.postimg.cc/xb8ckcy0z/t-sne_visulize_word_emb.png)](https://postimg.cc/image/if9tcrmm7/)

We want to find \\(e_w \approx e_{king} - e_{man} + e_{woman}\\). Our algorithm is thus:

\\[argmax_w \; sim(e_w, e_{king} - e_{man} + e_{woman})\\]

The most commonly used similarity function, \\(sim\\) is the _cosine similarity_:

\\[sim(u, v) = \frac{u^Tv}{||u||_2||v||_2}\\]

Which represents the **cosine** of the angle between the two vectors \\(u, v\\).

> Note that we can also use **euclidian distance**, although this is technically a measure of dissimilarity, so we need to take its negative. See [here](http://blog.christianperone.com/2013/09/machine-learning-cosine-similarity-for-vector-space-models-part-iii/) for an intuitive explanation of the cosine similarity measure.

### Introduction to word embeddings: Embedding matrix

Let's start to formalize the problem of learning a good word embedding. When we implement an algorithm to learn word embeddings, what we actually end up learning is an __embedding matrix__.

Say we are using a vocabulary \\(V\\) where \\(||V|| = 10,000\\). We want to learn an embedding matrix \\(E\\) of shape \\((300, 10000)\\) (i.e., the dimension of our word embeddings by the number of words in our vocabulary).

\\[E = \begin{bmatrix}e_{1, 1} & ... & e_{10000, 1}\\\\ ... & ... \\\\ e_{1, 300} & & ...\end{bmatrix}\\]

> Where \\(e_{i, j}\\) is the \\(j-th\\) feature in the \\(i-th\\) token.


Recall that we used the notation \\(o_i\\) to represent the one-hot encoded representation of the \\(i-th\\) word in our vocabulary.

\\[o_i = \begin{bmatrix}0 \\\ ... \\\ 1 \\\ ... \\\ 0\end{bmatrix}\\]

If we take \\(E \cdot o_i\\) then we are retrieving the embedding for the \\(i-th\\) word in \\(V\\), \\(e_i \in \mathbb R^{300 \times 1}\\).

#### Summary

 The  import thing to remember is that our goal will be to learn an **embedding matrix** \\(E\\). To do this, we initialize \\(E\\) randomly and learn all the parameters of this, say, 300 by 10,000 dimensional matrix. Finally, \\(E\\) multiplied by our one-hot vector \\(o_i\\) gives you the embedding vector for token \\(i\\), \\(e_i\\).

> Note that while this method of retrieving embeddings from the embedding matrix is intuitive, the matrix-vector multiplication is not efficient. In practice, we use a specialized function to lookup a column \\(i\\) of the matrix \\(E\\), an embedding \\(e_i\\).

### Learning word embeddings: Learning word embeddings

Lets begin to explore some concrete algorithms for learning word embeddings. In the history of deep learning as applied to learning word embeddings, people actually started off with relatively _complex_ algorithms. And then over time, researchers discovered they can use simpler and simpler algorithms and still get _very good_ results, especially for a large dataset. Some of the algorithms that are most popular today are so simple that they almost seem little bit magical. For this reason, it's actually easier to develop our intuition by introducing some of the more complex algorithms first.

> Note that a lot of the ideas from this lecture came from [Bengio et. al., 2003](http://jmlr.org/papers/volume3/bengio03a/bengio03a.pdf).

#### Algorithm 1

We will introduce an early algorithm for learning word embeddings, which was very successful, with an example. Lets say you are building a **neural language model**. We want to be able to predict the next word for any given sequence of words. For example:

> Note that, another common strategy is to pick a fixed window of words before the word we need to predict. The window size becomes a hyperparameter of the algorithm.

\\[x: \text{"I want a glass of orange ____"}\\]

One way to approach this problem is to lookup the embeddings for each word in the given sequence, and feed this to a densely connected layer which itself feeds to a single output unit with **softmax**.

[![neural_language_model_ex.png](https://s19.postimg.cc/f1k49mn1f/neural_language_model_ex.png)](https://postimg.cc/image/78tghnh27/)

Imagine our embeddings are 300 dimensions. Then our input layer is \\(\mathbb R^{6 \times 300}\\). Our dense layer and output softmax layer have their own parameters, \\(W^{[1]}, b^{[1]}\\) and \\(W^{[2]}, b^{[2]}\\). We can then use back-propagation to learn these parameters along with the embedding matrix. The reason this works is because the algorithm is incentivized to learn good word embeddings in order to generalize and perform well when predicting the next word in a sequence.

#### Generalizing

Imagine we wanted to learn the word "_juice_" in the following sentence:

\\[x: \text{"I want a glass of orange *juice* to go along with my cereal"}\\]

Typically, we would provide a neural language model with some _context_ and have it predict this missing word from that context. There are many choices here:

- \\(n\\) words on the left & right of the word to predict
- last \\(n\\) word before the word to predict
- a single, _nearby_ word

What researchers have noticed is that if your goal is to build a robust language model, choosing some \\(n\\) number of words before the target word as the context works best. However, if you goal is simply to learn word embeddings, then choosing other, simpler contexts (like a single, _nearby_ word) work quite well.

To summarize, by posing the language modeling problem in which some **context** (such as the last four words) is used to predict some **target** word, we can effectively learn the input word embeddings via backprogopogation.

### Learning Word Embeddings: Word2vec

In the last lecture, we used a neural language model in order to learn good word embeddings. Let's take a look at the the **Word2Vec** algorithm, which is a simpler and more computational efficient way to learn word embeddings.

> Most of the ideas in this lecture come from this paper: [Mikolov et al., 2013](https://arxiv.org/abs/1301.3781).

We are going to discuss the word2Vec **skip-gram** model for learning word embeddings.

#### Skip-gram

Let say we have the following example:

\\[x: \text{"I want a glass of orange juice to go along with my cereal"}\\]

In the skip-gram model, we choose (_context_, _target_) pairs in order to create the data needed for our supervised setting. To do this, for each _context_ word we randomly choose a _target_ word within some window (say, +/- 5 words).

_Our learning problem:_

\\[x \rightarrow y\\]
\\[\text{Context }, c\;(\text{"orange"}) \rightarrow \text{Target}, t\; (\text{"juice"})\\]

The learning problem then, is to choose the correct _target_ word within a window around the _context_ word. Clearly, this is a very challenging task. However, remember that the goal is _not_ to perform well on this prediction task, but to use the task along with backprogpogation to force the model to learn good word embeddings.

##### Model details

Lets take \\(||V|| = 10000\\). Our neural network involves an embedding layer, \\(E\\) followed by a softmax layer, similar to the one we saw in the previous lecture:

\\[E \cdot o_c \rightarrow e_c \rightarrow softmax \rightarrow \hat y\\]

Our softmax layer computes:

\\[p(t | c) = \frac{e^{\theta_t^Te_c}}{\sum^{10000}_{j=1}e^{\theta_j^Te_c}}\\]

> where \\(\theta_t\\) is the parameters associated with output \\(t\\) and the bias term has been omitted.

Which is a \\(|V|\\) dimensional vector containing the probability distribution of the target word being any word in the vocabulary for a given context word.

Our loss is the familiar negative log-likelihood:

\\[\ell (\hat y, y) = -\sum^{10000}_{i=1} y_i \log \hat y_i\\]

To _summarize_, our model looks up an embeddings in the embedding matrix which contains our word embeddings and is updated by backpropagation during learning. These embeddings are used by a softmax layer to predict a target word for a given context.

##### Problems with softmax classification

It turns out, there are a couple problems with the algorithm as we have described it, primarily due to the expensive computation of the _softmax_ layer. Recall our softmax calculation:

\\[p(t | c) = \frac{e^{\theta_t^Te_c}}{\sum^{10000}_{j=1}e^{\theta_j^Te_c}}\\]

Every time we wish to perform this softmax classification (that is, every step during training or testing), we need to perform a sum over \\(|V|\\) elements. This quickly becomes a problem when our vocabulary reaches sizes in the milllions or even billions.

One solution is to use a __hierarchial softmax__ classifier. The basic idea is to build a Huffman tree based on word frequencies. In this scheme, the number of computations to perform in the softmax layers scales as \\(\\log |V|\\) instead of \\(V\\).

> I don't really understand this.

##### How to sample the context c?

Sampling our target words, \\(t\\) is straightforward once we have sampled their context, \\(c\\), but how do we choose \\(c\\) itself?

Once solution is to sample uniform randomly. However, this leads to us choosing extremely common words (such as _the_, _a_, _of_, _and_, also known as stop words) much too often. This is a problem, as many updates would be made to \\(e_c\\) for these common words and many less updates would be made for less common words.

In practice, we use different heuristics to balance the sampling between very common and less common words.

#### Summary

In the original word2vec paper, you will find two versions of the word2vec model: the **skip-gram** one introduced here and another called **CBow**, the continuous bag-of-words model. This model takes the surrounding contexts from a middle word, and uses them to try to predict the middle word. Each model has its advantages and disadvantages.

The key problem with the **skip-gram** model as presented so far is that the softmax step is _very expensive_ to calculate because it sums over the entire vocabulary size.

### Learning Word Embeddings: Negative Sampling

In the last lecture, we saw how the **skip-gram** model allows you to construct a supervised learning task by mapping from contexts to targets, and how this in turn allows us to learn a useful word embeddings. The major the downside of this approach was that was the **softmax** objective was _very slow to compute_.

Lets take a look at a modified learning problem called **negative sampling**, which allows us to do something similar to the skip-gram model but with a much more efficient learning algorithm.

> Again, most of the ideas in this lecture come from this paper: [Mikolov et al., 2013](https://arxiv.org/abs/1301.3781).

Similar to the skip-gram model, we are going to create a supervised learning setting from unlabeled data. Explicitly, the problem is to predict whether or not a given pair of words is a  _context_, _target_ pair.

First, we need to generate training examples:

- **Positive** examples are generated exactly how we saw with the skip-gram model, i.e., by sampling a context word and choosing a target word within some window around the context.
- To generate the **negative examples**, we take a sampled context word and then for some \\(k\\) number of times, we choose a target word _randomly_ from our vocabulary (under the assumption that this random word _won't_ be associated with our sampled _context_ word).

As an example, take the following sentence:

\\[x: \text{"I want a glass of orange juice to go along with my cereal"}\\]

Then we might construct the following (context, target) training examples:

- \\((orange, juice, 1)\\)
- \\((orange, king, 0)\\)
- \\((orange, book, 0)\\)
- \\((orange, the, 0)\\)
- \\((orange, of, 0)\\)

Where 1 denotes a __positive__ example and 0 a __negative__ example.

> Note that this leads to an obvious problem: some of our randomly chosen target words in our generated negative examples will in fact be within the context of the sampled context word. It turns out this is OK, as much more often than not our generated negative examples are truly negative examples.

Next, we define a supervised learning problem, where our inputs \\(x\\) are these generated positive and negative examples, and our targets \\(y\\) are whether or not the input represents a true (_context_, _target_) pair (1) or not (0):

\\[x \rightarrow y\\]
\\[(context, target) \rightarrow 1 \text{ or } 0\\]

> Explicitly, we are asking the model to predict whether or not the two words came from a distribution generated by sampling from within a context (defined as some window around the words) or a distribution generated by sampling words from the vocabulary at random.

How do you choose \\(k\\)? Mikolov _et. al_ suggest \\(k=5-20\\) for small datasets, and \\(k=2-5\\) for large datasets. You can think of \\(k\\) as a 1:\\(k\\) ratio of **positive** to **negative** examples.

##### Model details

Recall the softmax classifier from the skip-gram model:

\\[\text{Softmax: } p(t | c) = \frac{e^{\theta_t^Te_c}}{\sum^{10000}_{j=1}e^{\theta_j^Te_c}}\\]

For our model which uses negative sampling, first define each input, output pair as \\(c, t\\) and \\(y\\) respectively. Then, we define a logistic regression classifier:

\\[p(y = 1 | c, t) = \sigma(\theta_t^Te_c)\\]

Where \\(\theta_t\\) represents the parameter vector for a possible target word \\(t\\), and \\(e_c\\) the embedding for each possible context word.

> NOTE: totally lost around the 7 min mark. Review this.

This technique is called __negative sampling__ because we generate our training data for the supervised learning setting by first creating a positive example and then _sampling_ \\(k\\) _negative_ examples.

##### Selecting negative examples

The final import point is how we _actually_ sample __negative__ examples in _practice_.

- One option is to sample the target word based on the empirical frequency of words in your training corpus. The problem of this solution is that we end up sampling many highly frequent stop words, such as "and", "of", "or", "but", etc.
- Another extreme is to sample the negative examples uniformly random. However, this also leads to a very non-representive sampling of target words.

What the [authors](https://arxiv.org/abs/1301.3781) found to work best is something in between:

\\[P(w_i) = \frac{f(w_i)^{\frac{3}{4}}}{\sum^{|V|}_{j=1}f(w_j)^{\frac{3}{4}}}\\]

Here, we sample proportional to the frequency of a word to the power of \\(\frac{3}{4}\\). This is somewhere between the two extremes of sampling words by their frequency and sampling words at uniform random.

#### Summary

To summarize,

- we've seen how to learn word vectors with a **softmax classier**, but it's very computationally expensive.
- we've seen that by changing from a softmax classification to a bunch of binary classification problems, we can very efficiently learn words vectors.
- as is the case in other areas of deep learning, there are open source implementations of the discussed algorithms you can use to learn these embeddings. There are also pre-trained word vectors that others have trained and released online under permissive licenses.

### Learning Word Embeddings: GloVe word vectors

The final algorithm we will look at for learning word embeddings is **global vectors for word representation** (__GloVe__). While not used as much as **word2vec** models, it has its enthusiasts -- in part because of its simplicity.

> This algorithm was original presented [here](http://www.aclweb.org/anthology/D14-1162).

Previously, we were sampling pairs of words (_context_, _target_) by picking two words that appear in close proximity to one another in our text corpus. In the GloVe algorithm, we define:

- \\(X_{ij}\\): the number of times word \\(i\\) appears in the context of word \\(j\\).
- \\(X_{ij}\\) == \\(X_{ji}\\)

> Note that \\(X_{ij}\\) == \\(X_{ji}\\) is not necessarily true in other algorithms (e.g., if we were to define the context as being the immediate next word). Notice that \\(i\\) and \\(j\\) play the role of \\(c\\) and \\(t\\).

#### Model

The models objective is as follows:

Minimize \\(\sum^{|V|}_{i=1} \sum^{|V|}_{j=1} f(X_{ij}) (\theta_i^Te_j + b_i + b_j' - \log X_{ij})^2\\)

- Think of \\(\theta_i^Te_j\\) as a measure of how similar two words are, based on how often the occur together: \\(\log X_{ij}\\). More specifically, we are trying to minimize this difference using gradient descent by searching for the pair of words \\(i, j\\) whose inner product \\(\theta_i^Te_j\\) is a good predictor of how often they are going to appear together, \\(\log X_{ij}\\).
- If \\(X_{ij} = 0\\), \\(\log X_{ij} = \log 0 = - \infty\\) which is undefined. We use \\(f(X_{ij})\\) as weighting term, which is 0 when \\(X_{ij}\\) = 0, so we don't sum over pairs of words \\(i, j\\) when \\(X_{ij} = 0\\). \\(f(X_{ij})\\) is also used to weight words, such that extremely common words don't "drown" out uncommon words. There are various heuristics for choosing \\(f(X_{ij})\\). You can look at the [original paper](http://www.aclweb.org/anthology/D14-1162) for details for how to choose this heuristic.

> Note, we use the convention \\(0 \log 0 = 0\\)

Something to note about this algorithm is that the roles of \\(theta\\) and \\(e\\) are now completely _symmetric_. So, \\(\theta_i\\) and \\(e_j\\) are symmetric in that, if you look at the math, they play pretty much the same role and you could reverse them or sort them around, and they actually end up with the same optimization objective. In fact, one way to train the algorithm is to initialize \\(\theta\\) and \\(e\\) both uniformly and use gradient descent to minimize its objective, and then when you're done for every word, to then take the average:

\\[e_w^{final} = \frac{e_w + \theta_w}{2}\\]

because \\(theta\\) and \\(e\\) in this particular formulation play symmetric roles unlike the earlier models we saw in the previous videos, where \\(theta\\) and \\(e\\) actually play different roles and couldn't just be averaged like that.

#### A note of the featurization view of word embeddings

Recall that when we first introduced the idea of word embeddings, we used a sort of _featurization view_ to motivate the reason why we learn word embeddings in the first place. We said, "Well, maybe the first dimension of the vector captures gender, the second, age...", so forth and so on.

However in practice, we cannot guarantee that the individual components of the embeddings are interpretable. Why? Lets say that there is some "space" where the first axis of the embedding vector is gender, and the second age. There is no way to guarantee that the actual dimension for each "feature" that the algorithm arrives at will be easily interpretable by humans. Indeed, if we consider the learned representation of each context, target pair, we note that:

\\[\theta_i^Te_j = (A\theta_i)^T(A^{-T}e_j) = \theta_i^TA^TA^{-T}e_j\\]

Where \\(A\\) is some arbitrary invertible matrix. The key take away is that the dimensions learned by the algorithm are not human interpretable, and each dimension typically encodes _some part_ of what we might think of a feature, as opposed to encoding an entire feature itself.

### Applications using Word Embeddings: Sentiment Classification

**Sentiment classification** is the task of looking at a piece of text and telling if someone likes or dislikes the thing they're talking about. It is one of the most important building blocks in NLP and is used in many applications. One of the challenges of sentiment classification is a lack of labeled data. However, with word embeddings, you're able to build good sentiment classifiers even with only modest-size label training sets. Lets look at an example:

\\[x: \text{"The dessert is excellent.", } y: 4/5 \text{ stars}\\]
\\[x: \text{"Service was quite slow.", } y: 2/5 \text{ stars}\\]
\\[x: \text{"Good for a quick meal, but nothing special.", } y: 3/5 \text{ stars}\\]
\\[x: \text{"Completely lacking in good taste, good service, and good ambience.", } y: 1/5 \text{ stars}\\]

> While we are using restaurant reviews as an example, sentiment analysis is often applied to [voice of the customer](http://www.wikiwand.com/en/Voice_of_the_customer) materials such as reviews and social media.

Common training set sizes for sentiment classification would be around 10,000 to 100,000 words. Given these small training set sizes, word embeddings can be extremely useful. Lets use the above examples to introduce a couple of different algorithms

#### Simple sentiment classification model

Take,

\\[x: \text{"The dessert is excellent.", } y: 4/5 \text{ stars}\\]

As usual, we map the tokens in the input examples to one-hot vectors, multiply this by a pre-trained embedding matrix and obtain our embeddings, \\(e_w\\). Using a pre-trained matrix is essentially a form of transfer learning, as we are able to encode information learned on a much larger corpus (say, 100B tokens) and use it for learning on a much smaller corpus (say, 10,000 tokens).

We could then _average_ or _sum_ these embeddings, and pass the result to a _softmax_ classifier which outputs \\(\hat y\\), the probability of the review being rated as 1, 2, 3, 4 or 5 stars.

This algorithm will work OK, but fails to capture _negation_ of positive words (as it does not take into account word order). For example:

\\[x: \text{"Completely lacking in good taste, good service, and good ambience.", } y: 1/5 stars\\]

might incorrectly be predicted to correspond with a high star rating, because of the appearance of "good" three times.

#### RNN sentiment classification model

A more sophisticated model involves using the embeddings as inputs to an RNN, which uses a softmax layer at the last timestep to predict a star rating:

[![Screen_Shot_2018-06-14_at_7.18.52_PM.png](https://s19.postimg.cc/kisjwkxrn/Screen_Shot_2018-06-14_at_7.18.52_PM.png)](https://postimg.cc/image/5zlev64mn/)

Recall that we actually saw this example when discussing many-to-one RNN architectures. Unlike the previous, simpler model, this model takes into account word order and performs much better on examples such as:

\\[x: \text{"Completely lacking in good taste, good service, and good ambience.", } y: 1/5 \text{ stars}\\]

which contain many negated, positive words. When paired with pre-trained word embeddings, this model works quite while.

#### Summary

Pre-trained word embeddings are especially useful for NLP tasks where we don't have a lot of training data. In this lecture, we motivated that idea by showing how pre-trained word embeddings can be used as inputs to very simple models to perform sentiment classification.

### Applications using Word Embeddings: Debiasing word embeddings

Machine learning and AI algorithms are increasingly trusted to help with, or to make, extremely important decisions. As such, we would like to make sure that, as much as possible, they're free of undesirable forms of bias, such as gender bias, ethnicity bias and so on. Lets take a look at reducing bias in word embeddings.

> Most of the ideas in this lecture came from this [paper](http://papers.nips.cc/paper/6228-man-is-to-computer-programmer-as-woman-is-to-homemaker-debiasing-word-embeddings.pdf).

When we first introduced the idea of word embeddings, we leaned heavily on the idea of analogical reasoning to build our intuition. For example, we were able to ask "Man is to woman as king is to ____?" and using word emebddings, arrive at the example queen. However, we can also ask other questions that reveal a _bias_ in embeddings. Take the following analogies encoding in some learned word embeddings:

\\[\text{"Man is to computer programmer as woman is to homemaker"}\\]
\\[\text{"Father is to doctor as mother is to nurse"}\\]

Clearly, these embeddings are encoding unfortunate gender stereotypes. Note that these are only examples, biases against ethnicity, age, sexual orientation, etc. can also become encoded by the learned word embeddings. In order for these biases to be learned by the model, they must first exist in the data used to train it.

#### Addressing bias in word embeddings

Lets say we have already learned 300D embeddings. We are going to stick to gender bias for simplicities sake. The process for debiasing these embeddings is as follows:

1 **Identify bias direction**:

Take a few examples where the only difference (or only major difference) between word embeddings is gender, and subtract them:
  - \\(e_{he} - e_{she}\\)
  - \\(e_{male} - e_{female}\\)
  - ...

Average the differences. The resulting vector encodes a 1D subspace that may be the __bias__ axis. The remaining 299 axes are the __non-bias direction__

> Note in the original paper, averaging is replaced by SVD, and the __bias__ axis is not necessarily 1D.

2 **Neutralize**:

For every word that is not definitional, project them onto __non-bias direction__ or axis to get rid of bias. These do __not__ include words that have a legitimate gender component, such as _"grandmother"_ but __do__ include words for which we want to eliminate a learned bias, such as _"doctor"_ or _"babysitter"_ (in this case a gender bias, but it could also be a sexual orientation bias, for example).

Choosing which words to neutralize is challenging. For example, _"beard"_ is characteristically male, so its likely not a good idea to neutralize it with respect to gender. The authors of the original paper actually trained a classifier to determine which words were definitional with respect to the bias (in our case gender). It turns out that english does not contain many words that are definitional with respect to gender.

3 **Equalize pairs**:

Take pairs of definitional words (such as _"grandmother"_ and _"grandfather"_ and equalize their difference to the __non-bias direction__ or axis. This ensures that these words are equidistant to all other words for which we have "neturalized" and encoded bias.

This process is a little complicated, but the end results is that these pairs of words, (e.g. _"grandmother"_ and _"grandfather"_) are moved to a pair of points that are equidistant from the __non-bias direction__ or axis.

It turns out, the number of these pairs is very small. It is quite feasible to pick this out by hand.

[![biased_embeddings.png](https://s19.postimg.cc/m4erkclpf/biased_embeddings.png)](https://postimg.cc/image/g3h2n9z33/)

#### Summary

Reducing or eliminating bias of our learning algorithms is a very important problem because these algorithms are being asked to help with or to make more and more important decisions in society. In this lecture we saw just one set of ideas for how to go about trying to address this problem, but this is still a very much an ongoing area of active research by many researchers.
