# Summary Examples

These examples are intented to recap and in some cases combine the methods that we've looked at. These examples are divided into two sets, that cover roughly the first and second half of the unit.

All of these examples are designed to be recapping methods we've already looked at, so with the possible exception of a bit of feature pre-processing everything in these examples should be familar. If it isn't, or if things are unclear, as always please ask questions in class, and perhaps go back and look at the content from the earlier weeks.

***

## Summary Week 1

### CAB420_Summary_1_Regression.ipynb

This example revists our regression methods:

*  Linear Regression
*  Regularised Regression using Ridge (L2 regularisation) and LASSO (L1 regularisation)
*  Deep Neural Networks

In this example we'll look at two datasets:

*  A dataset of diamond properties and prices
*  A dataset of songs

#### Diamonds

Here, we'll predict the price of a diamond given some properties. We'll see that on the surface at least, this works fairly well, but there are some issues.

#### Songs

Our second problem in this example is predicting the year of a song's release given some audio features. Essentially, we're regressing from audio features to a year. Sadly, as you'll see the data that doesn't work that well (this is also why we have two datasets). You're not going to see amazing regression results here, and that's ok. The poor performance we observe in this example is due to a few reasons including:

*  Weak correlation between the predictors and the response, i.e. we don't have much of a linear relationship to go off. This is a big problem, and one that it's kind of hard to come back from.
*  Reducing the dataset size, and dataset imbalance. To avoid long run-times, we're removing a lot of samples. We've also got some fairly massive imbalance in the data. We usually thing of imbalance in terms of classification, but it exists in regression too. Here, what you'll see if our dataset has a bit of an obsession with the late 90's and early 2000's, with most songs being from those years.

#### Things to Try

Using this example as a starting point, you may want to consider: 

*  Adding higher order terms to the diamond price task
*  Changing the data used in the song year prediction task to select a more balanced dataset (this will require downloading the entire dataset) 

### CAB420_Summary_2_Classification.ipynb

This example revists our classification methods:

*  K-Nearest Neighbours Classification
*  Support Vector Machines
*  Random Forests
*  Deep Neural Networks

In this summary example, we're going to use image data and throw that through all four of our methods. While feeding in raw images generally works really well for DCNNs, it's rarely a great choice for other methods. As such, we'll use a feature extraction process with these methods. One thing we won't do in this example is tune the classifiers or optimise them. Putting grid searches everywhere in this example is not going to be computationally efficient, so we'll avoid that, but please look at the second classification summary if you'd like see some grid searches cranking over. One thing we will do here however is have a look at run-times. We normally think of DCNNs as being the big slow behemoth of the machine learning world, so let's see if that holds up.

If you want to dig further into the feature extraction, feel free to ask questions, or check out the bonus example from the classification content which looks at HOG in more detail.

As from further activities, you may consider: 

*  Changing the HOG parameters and exploring how performance changes
*  Adding a grid search to optimise the non-deep learning methods
*  Using fine-tuning and/or augmentation to improve the deep learning performance 

### CAB420_Summary_3_Text_Classification.ipynb

This example revists our classification methods:

*  K-Nearest Neighbours Classification
*  Support Vector Machines
*  Random Forests
*  Deep Neural Networks

In this summary example, we're going to use text data, and in particular we're going to transform this into a Bag of Words representation. This is a new idea for CAB420 - though there is a Bonus example on this back with the classification content that you may have seen. There are a few problems that we may encouter with text data:

*  Each piece of text can be a different length, yet our methods all like to see data of the same length;
*  While order is important, the placement or a word of phrase is somewhat less so. The padding approach that we applied in our earlier multi-class classification example essentially places hard constraints on the placement of words.

Bag of Words will transform our variable length text into a fixed length representation, and also free us of this placement issue. As always, there's a cost and in this case it's the loss of any order information (though using n-grams we can capture local order, see the bonus example for details) and potentially (depending on our sequence length and other representation choices) a fairly large increase in dimensionality.

Using this Bag of Words (Bow) representation we will then:

*  Setup a hold-out validation set for use with a Grid Search
*  Train an SVM, CKNN and Random Forest using a Grid Search
*  Train a Deep Neural Network, operating over the same BoW data

Note also that in this summary I'm not going to go into any detail on model hyper-parameters. This is covered in a lot of detail in the first Classification summary, and I'd encourage you to review that if you're uncertain.

For some further exploration, you may wish to consider: 

*  Changing the Bag of Words parameters. You could change the thresholds for rare and/or common words, or extract n-grams and see how this impacts performance (and run-time). 

***
 
## Summary Week 2

### CAB420_Summary_4_PCA.ipynb

We've previously looked at PCA as pre-processing for classification tasks, but it doesn't have to be. The unsupervised nature of PCA means that it can be used as a pre-processing step for other methods such as regression. What's more, the fact that the set of principal components that PCA pulls out are orthogonal to each other has an interesting side-effect with regression. Let's explore.

If you're feeling curious, you may wish to: 

*  Add higher order terms to the regression model and pass all terms through PCA to see what happens. 


### CAB420_Summary_5_Learning_Representations.ipynb

We can think of a lot of the methods we've looked at in the second part of CAB420 as being concerned with learning a representation of the data.

PCA, LDA, Metric Learning methods, and Auto-Encoders all transform our data from it's original form to some lower dimensional version. The hope is that this lower dimensional version is in some way better for learning. This may be due to it simply being lower dimensional, and thus being easier to work with and helping to avoid overfitting, or perhaps the lower dimensional space removes noisy information that does not contribute to our overall objective (whatever that may be).

In this summary example, we're going to look at the impact that these various representation methods have on the performance of a simple classifier.

There's lot of other things you could try building on from this example. A good starting point would be: 

*  Explore the use of fine-tuning with multi-task and semi-supervised learning. Grab a pre-trained network (see https://keras.io/api/applications/Links to an external site. for example) and this as a backbone. 

### CAB420_Summary_6_Diarisation.ipynb

This example revisits our clustering mehods and explores an application of metric learning and clustering: Speaker Diarisation.

Speaker Diarisation is the problem of working out "who spoke when". A full blown speaker diarisation system has quite a few moving parts. A complete pipeline may include:

*  A speech activity detection module, that finds regions of speech in an audio recording
*  A speaker change detection module, that detects when the speaker changes (this component may or may not exist depending on the approach)
*  A speaker recognition engine, that extracts a vector (i.e. an embedding) to represent each section of speech
*  A clustering method, that groups segments of speech into those that are spoken by the same speaker

We're going to ignore the first step (and the optional second step) and take the segmentation information directly from the annotation. We'll address the second last part (speaker recognition) using metric learning where we can train a model to get embeddings for the speakers. For clustering we'll try both K-means and GMM (though we expect the GMM to work better).

Using this example as a starting point, youi are encouraged to:

*  Experiment with different values of K, or perhaps plot the BIC for a GMM as K varies for a given audio file. Does the BIC suggest the true value of K? 
*  Explore additional clustering methods. This is heading a bit out of scope (so if you don’t want to do this I understand), but HAC is a good choice of method in diarisation tasks. Using the bonus content as a guide, switch the clustering method across to HAC. 

