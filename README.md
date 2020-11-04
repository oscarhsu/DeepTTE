# DeepTTE

These are the code of AAAI 2018 paper ***When Will You Arrive? Estimating Travel Time Based on Deep Neural Networks***, but fixed some problems by OscarHsu.

This project provides the complete version of code and part of sample data in Chengdu.

Original: https://github.com/UrbComp/DeepTTE

# Environment :

This program run on python2.7 

**conda_env_CPU.yml** have the list of required libraries, include pyTorch-CPU.

**conda_env_GPU.yml** have the list of required libraries, include pyTorch-GPU.


# Usage:

```
python main_test_CPU.py
```
get the test results of DeepTTE by Pytorch-CPU version. The Deep Learning Model is prepared.


```
python main_test_GPU.py
```
get the test results of DeepTTE by Pytorch-GPU version. The Deep Learning Model is prepared.


```
python main_train_CPU.py
```
get the training model of DeepTTE by Pytorch-CPU version.


```
python main_train_GPU.py
```
get the training model of DeepTTE by Pytorch-GPU version.


# Data

In *** ./data/ ***, **testRemoveBeginLast** have the test data that is little bit different from original test data, **test**, the staying GPS points in the begin and the end of a trajectory are removed. 

**train_00**, **train_01**, **train_02**, **train_03**, **train_04** are 5-fold training data.

* testRemoveBeginLast_5 : test trajectories that each length is less then 5 km.
* testRemoveBeginLast_5_10 : test trajectories that each length is less then 10 km but greater then 5 km.
