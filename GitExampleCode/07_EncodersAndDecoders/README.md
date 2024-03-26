# Topic

## Core Examples

### CAB420_Encoders_and_Decoders_Example_1_AutoEncoders.ipynb

Autoencoders are an unsupervised neural network. They take an input, learn a compressed representation of that input, and then try to reconstruct it.

In that sense they function a little like PCA, but they can learn a highly non-linear representation. We can also "stack" them if we wish.

Autoencoders are rarely the end-product themselves, and so usually we'll train one and then modify or it in some other way to achieve our overall task. With that in mind, using this example as a starting point, try the following: 

*  Explore the use of an auto-encoder to pre-train a network for classification. To do this:
    *  Modify the autoencoder to have a more conventional structure, i.e., increasing number of filters as you go deeper into the encoder.
    *  Train the autoencoder.
    *  Truncate the trained autoencoder to remove the decoder (i.e., take the portion of the network from the input to the end of the decoder), and then append to this a classification head.
  *  Fine tune the network for classification. 
* Now, adapt your pre-trained auto-encoder to a semi-supervised setting. Borrowing from the ideas in the third example below (so go and look at that and then come back):
    *  Create an autoencoder as you did above. This time, at the middle of the network, attach a classification head so that you have your encoder, which feeds both the decoder and a classification head.
    *  Using the third example as a guide, mask out a hefty percentage of your data, and train the classification head using the semi-supervised categorical cross entropy function from the third example. 

### CAB420_Encoders_and_Decoders_Example_2_Multiple_Outputs.ipynb

In this example we will look at getting a network to do two things at once. Multi-task learning is generally very common with deep learning architectures. Largely because:

*  It's easy
*  It's often very useful
*  By having multiple outputs we often find that one task can regularise the other, i.e. the model is less likely to overfit and will generalise better. Both tasks don't always need to be super meaninful either, we can for example have one task that is simply an auto-encoder. This on it's own may not be of use to us, but it can help the other task that we're trying to learn depending on the circumstances.


### CAB420_Encoders_and_Decoders_Example_3_Semi_Supervised_Learning.ipynb

In this example we will look at how we can use semi-supervised learning to learn from partial data. Semi-supervised learning works best when you have two tasks, and typically these can be divided into:

*  One that is easy to annotate, and you have lots of data for;
*  One that is harder to annotate, and so you have less samples.

In my case I'm going to use a reduced classification task were I merge some classes; essentially I'll have a coarse task with full labels, and a fine-grained task with partial labels.

### CAB420_Encoders_and_Decoders_Example_4_VAE.ipynb

In this example we will look at a Variational Auto-Encoder. This will learn a latent representation, which we can then sample from to generate new samples, i.e. we can use our VAE to 'make up' new data.

*  Note, this has been adapted from the Keras VAE example [here](https://keras.io/examples/variational_autoencoder/).

The original VAE paper can be found here if you're interested:

*  [Kingma, Diederik P., and Max Welling. "Auto-Encoding Variational Bayes."](https://arxiv.org/abs/1312.6114)
*  Another good resource (that is slightly lighter reading) is [here](https://ermongroup.github.io/cs228-notes/extras/vae/), or this blog post [here](https://towardsdatascience.com/intuitively-understanding-variational-autoencoders-1bfe67eb5daf).

***

## Additional Examples


### CAB420_Encoders_and_Decoders_Additional_Example_Semantic_Segmentation.ipynb

In this example we'll look at how encoder-decoder structures can be used for semantic segmentation. We'll build a simple network ourselves, and then make it a bit less simple.

***

## Bonus Examples

### CAB420_Encoders_and_Decoders_Bonus_Example_GANs.ipynb

GANs are cool. But GANs are also fiddly, and confusing, hence they live in "bonus" teritory. Once upon a time they were in the main course content - in hindsight that was a poor choice.

In this example we will start to explore Generative Adversarial Networks, more commonly known as GANs. GANs are able to synthesise (i.e. make up) data, and have become very popular in machine learning for tasks where content creation is needed, or where mappings between domains (i.e. depth from monocular vision) are required. In this example we'll look at a couple of GANs that will learn to create images of clothing - basically we're training a fashion designer.

This example has been adapted from a few sources, including this [tensorflow example](https://www.tensorflow.org/tutorials/generative/dcgan) and the excellent, though increasingly old and out-of-date, [Keras GAN](https://github.com/eriklindernoren/Keras-GAN). I will add there are tons of GAN examples out there. I'm not trying to create the most amazing thing possible here - rather I'm looking for something compact enough to play with and serve as a starting point.

### CAB420_Encoders_and_Decoders_Bonus_Example_Self_Supervised_Learning.ipynb

As you can see from the title, this example is straddling a couple of different areas. We're taking the "self-supervised" idea that sits at the heart of an autoencoder, and throwing that up again our old friend, metric learning. What we saw with metric learning was that we could learn a nice, compact representation that put similar samples close to one another. This is very helpful for a lot of tasks, but does have that annoying old requirement that we need to have labels for each example. Self-Supervised learning get's around that.

We borrow the same idea that sits at the centre of the auto-encoder, that we can use the sample as it's own label, and apply this to a siamese network. In an auto-encoder, we exploit this idea by learning a network that will encode, and then decode, the same sample. The output should be identical to the input, but it goes via some stack of neural network layers which will (hopefully) learn some interesting representation we can use later. With a siamese network, we have two identical branches and the network is trained to produce an embedding that will have samples of the same close to each other. Here, rather than provide samples of the same (or different) classes, we simply provide two versions of the same image as our pair. To make the task non-trivial, we use augmentation to ensure that the images in the pair look suitably different.

This whole area of self-supervised learning is a really rapidly advancing area, and we're not going to try and cover all the really awesome stuff here, but we'll look at a few different methods that fit into this space. This example is heavily based off a few keras examples, in particular:

*  [Barlow Twins for Contrastive SSL](https://keras.io/examples/vision/barlow_twins/)
*  [Self-supervised contrastive learning with SimSiam](https://keras.io/examples/vision/simsiam/)

These have all been modified to be more consistent with each other, with what else we've done in CAB420, and to be a lot faster to execute (no large resnet models here). This means that performance will be limited, but if you have the time (and compute resources) you can scale up the backbone network and get some much better looking numbers.