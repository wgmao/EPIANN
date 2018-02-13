# EPIANN

Inspired by machine translation models we develp an attention-based nerual network model, **EPIANN**.
![Schematic overview of EPIANN](/EPIANN.png?raw=true)


## Data Augmentation
There are 6 cell lines. which are **celline**=**GM12878**, **HUVEC**, **HeLa-S3**, **IMR90**, **K562** and **NHEK**, and each comes with its own folder. Within each folder, there is a single file: **celline.csv**. **celline.csv** is a renamed copy of 
<p align="center">
https://<i></i>github.com/shwhalen/targetfinder/tree/master/paper/targetfinder/<b>celline</b>/output-ep/pairs.csv
</p>

Before we actually train oorneural network model, we need to generate input data from genomic coordinates(hg19) of enhancers and promoters, along with the indicators of EPIs recorded in **celline.csv**. **Data_Augmentation.R** encoded an automatic data augmentation pipeline with several parameters specified in the following table.

Parameters| Explanation
--- | ---
celline| change it to one of the 6 cell lines with default = "IMR90"
folder | the name of the folder to hold all output files with default = "aug_50"
shift_distance  | the step size to slide extended region around the enhancer and promoter with default = 50
enhancer_target_length| the length of extended enhancer with default = 3000
promoter_target_length| the length of extended promoter with default = 2000 
positive_scalar| the augmentation ratio with default = 20
test_percent| the percent of test data among all with default = 0.1
random_seed| the random seed to sample test data with default = 1

You can find the output files with default parameters under the directory IMR90/aug_50/. ~~The following files are currently not avaiable in the github repository because of the size limit (Work In Progress).~~ They are avaiable in the repository.

```
IMR90/aug_50/IMR90_enhancer.fasta
IMR90/aug_50/IMR90_promoter.fasta
IMR90/aug_50/imbalanced/IMR90_enhancer.fasta
IMR90/aug_50/imbalanced/IMR90_promoter.fasta
```


## Train Neural Netork Model
Under the directory IMR90/, you can find an example python script **IMR90_EPIANN.py** with the default setting. The parameters regarding inputs are explained in the following table.

Parameters| Explanation
--- | ---
celline|chanage it to one of the 6 cell lines with default = 'IMR90'
file_pre|change it to be the folder containing augmented data with default = 'aug_50/IMR90'
out_dir|change it to be the folder that contains the output with dedault = 'output/IMR90_EPIANN'
script_id|change it to be the current python script name in order to distinguish the outputs from multiple runs with default = 'IMR90_EPIANN'

The computational grpaph for the neural network is programmed using Tensorflow. On our setup, we use a single NVIDIA GTX 1080 or NVIDIA TITAN X with 5 CPU threads. A single batch takes about 6 seconds to train. All neural neural parameters can be altered in the script.

Neural Network Parameters| Explanation
--- | ---
enhancer_length| the length of input enhancers with default = 3000
promoter_length | the length of input promoters with default = 2000
BATCH_SIZE | the half of exact batch size with default = 32
num_filters| the number of convolution filters with default = 256
e_conv_width| the convolutional filter width with default = 15
dropout_rate_cnn| the dropout rate for the convolution layer with default = 0.2
dropout_rate| the dropout rate for all layers except the convolution layer with default = 0.2
pool_width| the max pooling size with default = 30
atten_hyper| the dimension of the attention-related parameters with default = 32
dense_neuron_coor| the dimension of the fully connected layers for *coordinate prediction* with default = [128, 64]
inter_dim| the dimension of the interaction quantification related parameters with default = 1
topk| the top-k pooling size with default = 32
dense_neuron| the dimension of the fully connected layers with default = 32
lamb| the hyperparameter which mediates the cross-entropy error and the regression error with default = 10
num_of_epoch| the number of epochs with default = 90
output_step| the step size to report performance on test dataset with default = 500 batches


## Required Pre-installed Packages
R (3.4.2) Library dependencies

```
GenomicRanges 1.28.2
BSgenome.Hsapiens.UCSC.hg19.masked 1.3.99
```

Python (2.7.6) Module dependencies
~~Sklearn 0.18.1~~
```
os
pickle
time
tensorflow 1.3.0
numpy 1.13.3
Sklearn 0.19.1
Biopython 1.67
```
