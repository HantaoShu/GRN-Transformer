# GRNTransformer

## About
This directory contains the code and resources of the following paper:
<i> TAB </i>

## Overview of the Model

<p align="center">
<img  src="fig/Figure_1.jpg" width="800" height="400" > 
</p>

## Dependencies
- python==3.7
- pytorch==1.7.0+cu101
- pandas==1.1.4
- numpy==1.19.4
- scikit-learn==0.23.2
- argparse==1.4.0

All dependencies can be installed within a few minutes.

## Description of directories and files
- [src] contains the detail implementation for self-supervised step of GRNTransformer
- [demo_data] contains the example of used dataset. 
    - BEELINE
        - 500_ChIP-seq_mDC
            - inputs Including the gene expression data 'data.csv', ground truth cell-type-specific GRN 'label.csv', 
            GRN inference result by using PIDC 'PIDC_output_file.txt', train-test-split index files 'split_non_*.pkl', 
            TF-target names 'train_z.npy', ground truth cell-type-specific GRN labels 'train_y.npy', and 
            cell-type-non-specific ChIP-seq training labels 'train_y_non.npy'.
            - outputs Including example pretrain result 'pretrain_output.pkl' by running code self_supervised_main.py 
            and 
            example performance output file 'performance.pkl' by running code supervised_main.py
        - other_data Including TFs and cell-type-non-specific GRN collected by BEELINE benchmark.
    - Simulation 
        - ER/SF
            - inputs  Similar with inputs directory of 500_ChIP-seq_mDC except train-test-split index files. 
            'split\_\*\_0.5.pkl' denotes to the split file for train-test-split for random-split setting and 
            split\_TF\_\*\_0.5
            .pkl' 
            denotes to the split file for split-TF setting.
            - outputs Similar with outputs directory of 500_ChIP-seq_mDC 

- [self_superivsed_main.py]  The main function for self-supervised learning step.
- [supervised_main.py] The main function for supervised learning step.

## Usage
We take 500_ChIP-seq_mDC data as example.

- self-supervised step
    ```sh
    python self_superivsed_main.py --data_file ./demo_data/BEELINE/500_ChIP-seq_mDC/inputs/data.csv --PIDC_file 
    ./demo_data/BEELINE/500_ChIP-seq_mDC/inputs/PIDC_output_file.txt --save_name pretrain_output.pkl
    ```
    where '--data_file' denotes to the input expression data '--PIDC_file' denotes to the input pre-calculated PIDC 
    file, '--save_name' denotes to the output filename which is used in following supervised step.
    
- supervised step 
    ```sh
    python supervised_main.py --split_file split_non_1.pkl  --pre_GRN_file 
    ./demo_data/BEELINE/500_ChIP-seq_mDC/outputs/pretrain_output.pkl  --data_dir 
    ./demo_data/BEELINE/500_ChIP-seq_mDC/inputs/ --train_y_file  train_y_non.npy --output_file 
    ./demo_data/BEELINE/500_ChIP-seq_mDC/outputs/performance.pkl
    ```
  where '--pre_GRN_file' denotes to the output file of previous self-supervised step, '--data_dir' denotes to the 
  directory which include all training data, '--train_y_file' denotes to the name of training label file, and 
  '--output_file' denotes to the output file name.
 
If you have any question, please feel free to contact to me. \
Email: sht18@mails.tsinghua.edu.cn, shuht96@gmail.com
 
 # Other baseline single-cell RNA-seq package 
 - BEELINE https://github.com/Murali-group/Beeline
 - CNNC https://github.com/xiaoyeye/CNNC
 - DeepDRIM https://github.com/jiaxchen2-c/DeepDRIM