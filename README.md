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
