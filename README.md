# Workshop on machine learning for Physis September 2024

This is a very basic site for some background for the workshop at the
Physis pedagogical forum 2024. Given the relatively short notice -
the procedure below is not double-checked now but it did work in
February 2024.


## Slides

There was not enough time to present even a fraction of the slides
prepared, so here they are for your perusal:
[Slides (note:380Mb!)](https://www.dropbox.com/scl/fi/eacadq12e8p9mymto370v/workshop.pdf?rlkey=jtvabce3y71uxt47z8j65y06v&dl=0).

The file is fairly large because of the big images in it. If you are
happy to skip the Euclid images, then [this version](https://www.dropbox.com/scl/fi/gdupjp9madbg0iqd271ii/workshop-noimages.pdf?rlkey=536u2sbkadmzj1nhs2s0agl48&dl=0)
is only 46Mb.


## Software you need for the workshop

The workshop will make use of python, and for this you need a recent
version of python installed. I use python 3 by default and while some
scripts will work for python 2, there is really no good reason for
continuing to use python 2 (with some exception for important legacy
code). For python you will need (well, I recommend it at least) at
least these libraries installed:

- [numpy](http://www.numpy.org/) - for numerical calculations
- [astropy](http://www.astropy.org/) - because we are astronomers
- [scipy](https://www.scipy.org/) - because we are scientists 
- [sklearn](https://scikit-learn.org/) - Machine learning libraries with full name scikit-learn.
- [matplotlib](https://matplotlib.org/) - plotting (you can use alternatives of course)
- [pandas](https://pandas.pydata.org/) - nice handling of data
- [seaborn](https://seaborn.pydata.org/) - nice plots

(the last two are really "nice to have" but if you can install the
others then these are easy). 

Personally I use the [Mamba](https://mamba.readthedocs.io/en/latest/)
python package manager, but I used to use the
[Anaconda](https://www.anaconda.com/products/distribution) Python
distribution. Either can be used to manage one's python installation
and to create environments. I strongly recommend using environments
(often called virtual environments) for this course. These come in two
main flavours, the built-in `venv` virtual environments, or the ones
provided by `conda` or `mamba`. See for instance [this
overview](https://www.machinelearningplus.com/deployment/conda-create-environment-and-everything-you-need-to-know-to-manage-conda-virtual-environment/)
for instance (which is focused on `conda` - `mamba` works pretty much
the same) or [this
one](https://realpython.com/python-virtual-environments-a-primer/) for
a more `venv` focused intro. Since I use `conda` or `mamba` (they are
totally exchangeable) my examples will use that but it is pretty easy
to translate to `venv` instead.

To set things up for this course, what I did (after installing
`anaconda` or `mamba`) was
```
# Create an environment
> conda create -n mld2024 numpy scipy scikit-learn pandas seaborn matplotlib jupyter astropy pip
...
> conda activate mld2024

# or using mamba:
> mamba create -n mld2024 numpy scipy scikit-learn pandas seaborn matplotlib jupyter astropy pip
...
> mamba activate mld2024  # depending on your shell you might have to
use conda activate instead

```
The first command is done only once, the second is done every time you
start a new shell. 


You should also get `astroML` which has a nice web page at
[http://www.astroml.org/](http://www.astroml.org/) and a git
repository at
[https://github.com/astroML/astroML](https://github.com/astroML/astroML). This
is the website associated to the "Statistics, Data Mining, and Machine
Learning in Astronomy" book mentioned above. They also provide clear
[installation
instructions](http://www.astroml.org/user_guide/installation.html). Personally
I used their "From Source" instructions but it is probably in general
easier to use the "Conda" instructions if you use Anaconda and the
"Python Package Index" instructions otherwise. 




# Getting ready for deep learning in python

Deep learning is what a lot of people think of when they think about
machine learning. I hope that I will be able to make clear in my
workshop that this is not the case and you need a wide range of tools
to do the best you can.

But if I find time, we will look at using deep learning in python at
the end of the workshop. In order to follow the examples, you will
need to have some software installed. This is more involved than what
we had above so might take some time to get working.


There are quite a few libraries for this around but we will
use the most commonly used one,
[TensorFlow](https://www.tensorflow.org/) and we will use the
[keras](https://keras.io/) python package for interacting with
TensorFlow. Keras is a high-level interface (and can also use other
libraries, [Theano](https://github.com/Theano/Theano) and
[CNTK](https://github.com/Microsoft/cntk), in addition to TensorFlow).

In addition, since there was no time to prepare properly, we will look
at another example created by John Wu that uses fastai. These
notebooks can be found at 
[John F. Wu's resources
site](https://jwuphysics.github.io/resources/). We will look at the
notebooks from the 2022 Astro Hack Week, but there are many other
interesting resources there.


There are many pages that detail the installation of these packages
and what you need for them. A good one with a bias towards Windows is
[this
one](https://towardsdatascience.com/setup-an-environment-for-machine-learning-and-deep-learning-with-anaconda-in-windows-5d7134a3db10). I
will give a very brief summary here of how I set things up. This is
not optimised for Graphical Processing Unit (GPU) work so for serious
future work you will need to adjust this.

### Create an environment in anaconda 

I am going to assume you use anaconda for your python environment. If
not, you need to change this section a bit - use virtualenv instead of
setting up a conda environment.  It is definitely better to keep your
TensorFlow/keras etc setup out of your default Python work
environment. Most of the packages are also installed with `pip` rather
than conda, so what I use in this case for tensorflow is:

```
> conda create -n tensorflow pip
<...>
> activate tensorflow
```
It is important to activate the environment, otherwise you'll mess up
your default conda environment!  Check that your prompt says
`[tensorflow]` before continuing (see below):

```
[tensorflow] > pip install matplotlib astropy pandas scikit-learn seaborn jupyter astroML
<...>
[tensorflow] > 
```

For fastai I can use conda throughout (well, I use mamba, but it is
exactly the same):
```
mamba create -n fastai jupyter matplotlib astropy numpy scikit-learn
seaborn fastai
```
creates an environment `fastai` that you can use. Nothing more needed
for fastai, but for tensorflow we still need some steps:



### Install tensorflow and keras

Tensorflow is a large package - 244 Mb in my installation and it
requires a fair number of additional packages so this can take a bit
of time. 

```
[tensorflow] > pip install --upgrade tensorflow
<...>
[tensorflow] > pip install --upgrade keras
```

That should set up you fairly well for a first dip into deep learning. 


Unfortunately this might not work for you out of the box - in
particular the dependency of `tensorflow` on `grpcio` can lead to a
lot of problems on Mac OS. See [this
discussion](https://github.com/grpc/grpc/issues/30723) for some
discussion of this. What I ended up doing was installing `grpcio` via
`conda`. 



# Some additional information

## Getting hold a more complete course

I usually teach machine learning in astronomy as a course at UP in the
doctoral program (it does not rely on very advanced knowledge so is
perfectly accessible at a BSc/MSc level too). This more extensive
course you can find at [MLD2024 repo](https://github.com/jbrinchmann/MLD2024). 


## Literature for learning more


Below you can find some books of use. The links from the titles get
you to the Amazon page. If there are free versions of the books
legally available online, I include a link as well.

- I base myself partially on ["Statistics, Data Mining, and Machine Learning in Astronomy" - Ivezic, Connolly, VanderPlas &amp; Gray](http://www.amazon.co.uk/Statistics-Mining-Machine-Learning-Astronomy/dp/0691151687/ref=sr_1_1?ie=UTF8&amp;qid=1444255176&amp;sr=8-1&amp;keywords=Statistics%2C+Data+Mining%2C+and+Machine+Learning+in+Astronomy+-+Ivezic%2C+Connolly%2C+VanderPlas+%26+Gray)

- I have also consulted ["Deep Learning" - Goodfellow, Bengio &amp; Courville](https://www.amazon.co.uk/Deep-Learning-Adaptive-Computation-Machine/dp/0262035618/ref=sr_1_1?ie=UTF8&amp;qid=1505297517&amp;sr=8-1&amp;keywords=Deep+Learning)

- ["Pattern Classification" - Duda, Hart &amp; Stork](http://www.amazon.co.uk/Pattern-Classification-Second-Wiley-Interscience-publication/dp/0471056693/ref=sr_1_1?ie=UTF8&amp;qid=1444255264&amp;sr=8-1&amp;keywords=Pattern+Classification), is a classic in the field

- ["Pattern Recognition and Machine Learning" - Bishop](http://www.amazon.co.uk/Pattern-Recognition-Machine-Learning-BISHOP/dp/8132209060/ref=sr_1_1?ie=UTF8&amp;qid=1444255326&amp;sr=8-1&amp;keywords=Pattern+Recognition+and+Machine+Learning+-+Bishop), is a very good and comprehensive book. Personally I really like this one.

- ["Bayesian Data Analysis" - Gelman](http://www.amazon.co.uk/Bayesian-Analysis-Chapman-Statistical-Science/dp/1439840954/ref=sr_1_1?ie=UTF8&amp;qid=1444255416&amp;sr=8-1&amp;keywords=Bayesian+Data+Analysis+-+Gelman), is often the first book you are pointed to if you ask questions about Bayesian analysis.

- ["Information Theory, Inference and Learning Algorithms" - MacKay](http://www.amazon.co.uk/Information-Theory-Inference-Learning-Algorithms/dp/0521642981/ref=sr_1_1?ie=UTF8&amp;qid=1444255466&amp;sr=8-1&amp;keywords=Information+Theory%2C+Inference+and+Learning+Algorithms), is a very readable book on a lot of related topics. The book is also [freely available](http://www.inference.phy.cam.ac.uk/itila/book.html) on the web.

- ["Introduction to Statistical Learning - James et al"](http://www.amazon.co.uk/Introduction-Statistical-Learning-Applications-Statistics/dp/1461471370/ref=sr_1_fkmr0_1?ie=UTF8&amp;qid=1444255565&amp;sr=8-1-fkmr0&amp;keywords=Introduction+to+Statistical+Learning+-+James+et+al) is a readable introduction (fairly basic) to statistical technique of relevance. It is also [freely available](http://www-bcf.usc.edu/~gareth/ISL/) on the web.

-["Elements of Statistical Learning - Hastie et al](http://www.amazon.co.uk/Elements-Statistical-Learning-Prediction-Statistics/dp/0387848576/ref=sr_1_1?ie=UTF8&amp;qid=1444255710&amp;sr=8-1&amp;keywords=Elements+of+Statistical+Learning), is a  more advanced version of the Introduction to Statistical Learning with much the same authors. This is also [freely available](http://statweb.stanford.edu/~tibs/ElemStatLearn/) on the web.

- ["Bayesian Models for Astrophysical Data", Hilbe, Souza & Ishida](https://www.amazon.com/Bayesian-Models-Astrophysical-Data-Python/dp/1107133084) is a good reference book for a range of Bayesian techniques and is a good way to learn about different modelling frameworks for Bayesian inference. 


