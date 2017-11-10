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
num_of_trails | number of random searches before GP starts 
num_of_GP     |number of Gaussian Process search steps     
p_upper_limit| p is between -1 and 1 and this is set to be 1 most of time
nPC| upper bound on the number of principal components
p_thresh| 0.05 is standard
svdres|output from svd() function

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
