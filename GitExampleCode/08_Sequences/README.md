# Sequences

## Core Examples

### CAB420_Sequences_Example_1_Recurrent_Neural_Networks.ipynb 

We've actually looked at a couple of methods for processing sequences already, though we didn't explicitly look at these in the context of working with sequential data. In summary:

*  In week 3 when looking at classification, CAB420_Classification_Example_2_Multi_Class_Classification.ipynb explored classifying beer names. These are sequences of words, and to get this into a format we could work with we:
    *  Used a word embedding model to tranform each word to a vector of some fixed size;
    *  Concatenated all words in a beer name to obtain a feature;
    *  Padded these features up to some length, and removed any long names, to ensure all beer names have the same length representation.
*  In week 6 in the summary example CAB420_Summary_3_Text_Classification.ipynb, we used Bag-of-Words to transform our variable length phrase (in this case a tweet) into a fixed length histogram.

These two approaches haven't been that great in terms of performance, but perhaps more critically they also haven't really used/modelled the data as a "sequence". We'll now play with a recurrent neural networks, which will actually recieve and treat the data as a sequence.

### CAB420_Sequences_Example_2_Sequence_to_Sequence_Prediction.ipynb 

Often when dealing with sequences, often we wish to predict a sequence as our output. We can easily use neural networks to do that as well.

Using examples this example as a basis, while also borrowing from CAB420_Neural_Attention_Example_1_Attention_and_RNNs.ipynb, try the following: 

*  Add a simple attention mechanism to the sequence-to-sequence prediction task. Explore what happens when you modify some of the operations of the attention mechanism, such as:
    *  Switching between a softmax and sigmoid activation on the attention scores.
    *  Using additive rather than multiplicative attention. 

***

## Additional Examples

### CAB420_Sequences_Additional_Example_Sequence_Classification.ipynb

WARNING: This example is not quick. Training of the SVM in particular will take a while.

Sequences are everywhere, but they need to be handled differently from the tabular data or the sort of image or audio data that we've used so far. The big differences are that:

*  sequences can be of different lengths, and
*  sequences often have some inherent ordering going on that ideally, we'd like take notice of and deal with appropriately. 

In this example we'll look at a really simple way to work with sequences by creating a fixed length representation.

This approach is rarely optimal, but it's easy to apply and allows us to use sequeces with all the methods that we've already looked at.

***

## Bonus Examples

### CAB420_Sequences_Bonus_Example_Bag_of_Words_and_Latent_Dirichlet_Allocation.ipynb

In a previous bonus example, and in a summary example, we used bag-of-words to transform a variable length sequence into a fixed length representation. 

Bag of Words creates a histogram for each sequence, where each bin of the histogram corresponds to a word. As such, we go from a variable length document to a fixed length histogram where the length is controlled by the size of dictionary. While this sounds like a method that can only be used for language, we can create analogues for words in other domains, and as such this method is used quite broadly. One limitation of this however if that by building a histogram, we totally destroy any information on the order of the data.

Once we have this representation, there are other things we can do with it. We can:

*  learn directly from it, treating it as the feature for our classifier
*  pass it through another process, extracting a more compact representation

One popular family of methods used with Bag of Words is topic models. These model a document as a distribution of topics. A topic is a distribution of words. The topics themselves are learnt from the data, This means that we can then represent a sample not by the words that are in it, but by the topics that the doument contains. The method that we'll use for this is Latent Dirichlet Allocation (the other LDA), and this is, in essence, a clustering technique which aims to learn the topics from a large corpus of data. Like some of our other clustering techniques, it requires us to specify the number of topics (or clusters) up front.
