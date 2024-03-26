# Neural Attention

## Core Examples

### CAB420_Neural_Attention_Example_1_Attention_and_RNNs.ipynb

The idea of attention is to learn which parts of the input (or of some internal representation) are most relevant, and to try to direct the network towards these parts. There are very many different variations on attention, but the general approach can be summarised as:

*  Learn an additional layer that will consider the importance of locations within an input
*  Use this learned importance to weight the input/representation such that more important regions are more prominent

Attention actually first emerged in text processing, and this example will quickly explore how we can use a simple attention layer within an LSTM network for classifying tweets.

### CAB420_Neural_Attention_Example_2_Transformers.ipynb

Transformers are a specific class of feed-foward network that achieve state of the art performance for sequence processing tasks, and dominante Natural Language Processing (NLP). Unlike recurrent models which store an internal memory to keep track of data over the sequence, transformers do not use memory or recurrent steps, and process the entire sequence at once. One of the main motivations for this to increase the parallelisation of the networks, allowing them to run faster in large (and expensive) GPU configurations. Attention is used liberally to help the model understand connections and relationships in the data.

For most tasks, multiple transformer units are stacked, allowing information to be extracted by successive passes through attention blocks. For sequence to sequence tasks, transformers are used to encode and then decode the representation, much like we've seen in auto-encoder or other image-to-image networks.

This example will keep things simple, and will:

*  Build a simple transformer block
*  Use this to complete out twitter sentiment classification task Critically, we won't look the decoder stage which is used in applications such as language translation.

For those who are interested in further reading regarding transformers, the following are good starting places:

*  [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
*  [Tensor 2 Tensor Notebook (contains cool visualisation)](https://colab.research.google.com/github/tensorflow/tensor2tensor/blob/master/tensor2tensor/notebooks/hello_t2t.ipynb)
*  [Attention is all you need (the paper that introduced this architecture)](https://arxiv.org/abs/1706.03762)

To clarify, thankfully Michael Bay has no involvement what so ever in this example (although a few gratuitous explosions would be fun).

As an activity, take this example and the sequence-to-sequence problem back in CAB420_Sequences_Example_2_Sequence_to_Sequence_Prediction.ipynb and try the following: 

*  Use a transformer for the sequence-to-sequence task shown in CAB420_Sequences_Example_2_Sequence_to_Sequence_Prediction.ipynb.  


***

## Additional Examples

***

## Bonus Examples

### CAB420_Neural_Attention_Bonus_Example_Attention_and_DCNNs.ipynb

Attention is a network component that has become very common in many state of the art deep neural networks. We already saw

The idea of attention is to learn which parts of the input (or of some internal representation) are most relevant, and to try to direct the network towards these parts. There are very many different variations on attention, but the general approach can be summarised as:

* Learn an additional layer that will consider the importance of locations within an input
* Use this learned importance to weight the input/representation such that more important regions are more prominent

We'll look at a few variations on this these within the context of images in this example.

### CAB420_Neural_Attention_Bonus_Example_Visual_Transformer.ipynb

We've seen transformers used for text processing, which is what they were originally proposed for. However, they are becomming increasingly common for other tasks, including vision tasks.

The challenge in using a transformer for a vision task, in particular a task which operates over just a single image, is that the data isn't really a sequence - but we can make it one. By splitting an image into a sequence of patches, and obtaining an embedding per patch, we can get a representation that can be fed to a transformer. By virtue of the position embedding, we even retain some knowledge about where in the image the patch came from, allowing us to still leverage spatial information. 

It's worth considering the different between the spatial information captured by the transformer vs what happens with a DCNN. With a DCNN, our receptive field get's progressivley bigger as we go deeper into the network. Filters in shallow layers only see a very small part of the image, but by the time we get to the deeper layers the filter is effectivley able to see most (or even all) of the input frame.

With a transformer, the multi-head attention mechanism looks at each element in relation to all others. If each of our elements represents a small patch, the transformer layer will look at how each patch relates to each other patch. We can (kind-of) view this as the transformer layer having a receptive field across the entire image, and so there is the potential to capture relationships between distant spatial regions in much shallower networks.

At this we'll stop with the text and get into the code; though please note that this example is adapted from [here](https://keras.io/examples/vision/image_classification_with_vision_transformer/).