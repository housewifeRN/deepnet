############################
MULTIMODAL DEEP BOLTZMANN MACHINES

Nitish Srivastava
University of Toronto
############################
This code trains a Multimodal DBM on the MIR-Flickr dataset.
The implementation uses GPUs to accelerate training.

(1) GET DATA
  - Download preprocessed data into a place with lots of disk space.
  $ mkdir -p path/to/data
  $ cd path/to/data
  $ wget http://www.cs.toronto.edu/~nitish/multimodal/flickr_data.tar.gz
  $ tar -xvzf flickr_data.tar.gz 

(2) TRAIN MULTIMODAL DBN
  - Change to the directory containing this file.
  - Edit paths in runall_dbm.sh
  - Train dbm
  $ ./runall_dbm.sh

This implementation has not been tested.
