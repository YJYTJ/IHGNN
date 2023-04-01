# IHGNN
## About
PyTorch implementation of IHGNN.
## Installation
This implementation is based on the structure2vec graph backend. Under the "lib/" directory, type
```
make -j4
```
to compile the necessary c++ files.

After that, under the root directory of this repository, type
```
python main.py -data DATASETNAME
```
for datasets that have node labels.

And type
```
python main.py -data DATASETNAME -degree_as_tag 1
```
for datasets that don't have node labels, such as COLLAB.

To see the results, type
```
./compute_acc.sh
```
## Datasets
Default graph datasets are stored in "data/DATASETNAME/DATASETNAME.txt". Check the "data/README.md" for the format.
