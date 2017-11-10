# EPIANN

Inspired by machine translation models we develp an attention-based nerual network model, **EPIANN**.
![Schematic overview of EPIANN](/EPIANN.png?raw=true)


## Data Augmentation
There are 6 cell lines. which are **celline**=**GM12878**, **HUVEC**, **HeLa-S3**, **IMR90**, **K562** and **NHEK**, and each comes with its own folder. Within each folder, there is a single file: **celline.csv**. **celline.csv** is a renamed copy of 
<p align="center">
https://<i></i>github.com/shwhalen/targetfinder/tree/master/paper/targetfinder/<b>celline</b>/output-ep/pairs.csv
</p>

Parameter| Explanation
--- | ---
celline| change it to one of the 6 cell lines with default = "IMR90"
folder | the name of the folder to hold all output files with default = "aug_50"
shift_distance  | the step size to slide extended region around the enhancer and promoter with default = 50
enhancer_target_length| the length of extended enhancer with default = 3000
promoter_target_length| the length of extended promoter with default = 2000 
positive_scalar| the augmentation ratio with default = 20
test_percent| the percent of test data among all with default = 0.1
random_seed| the random seed to sample test data with default = 1

## Train Neural Netork Model

Parameter| Explanation
--- | ---
celline|chanage it to one of the 6 cell lines with default = 'IMR90'
file_pre|change it to be the folder containing augmented data with default = 'aug_50/IMR90'
out_dir|change it to be the folder that contains the output with dedault = 'output/IMR90_EPIANN'
script_id|change it to be the current python script name in order to distinguish the outputs from multiple runs with default = 'IMR90_EPIANN'


Neural Network Parameter| Explanation
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
```
os
pickle
time
tensorflow 1.3.0
numpy 1.13.3
Sklearn 0.18.1
Biopython 1.67


```
