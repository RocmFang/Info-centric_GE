<meta name="robots" content="noindex">

## FeLoG
This codebase gives the reference implementation of FeLoG



# Prerequisites

- Ubuntu 20.04
- Linux kernel 5.4.0
- g++ 9.4.0
- CMake 3.10.2
- CUDA 12.2
- [MPICH 3.4.2](https://www.mpich.org)
- [MKL 2022.0.2](https://software.intel.com/en-us/mkl)

# Datasets

The evaluated dataset Youtube and LiveJournal are prepraed in the "dataset" directory.

Since the the space limited of the repository, the other datasets [Twitter](https://law.di.unimi.it/datasets.php), [Com-Orkut](https://snap.stanford.edu/), [Flickr](http://datasets.syr.edu/pages/datasets.html)，[U.K.-2007](https://law.di.unimi.it/webdata/uk-2007-05/) and [OGB-papers100M](https://snap.stanford.edu/ogb/data/nodeproppred/) can be found in their open resource.

# Setup

First Compile FeLoG with CMake:

```
mkdir build && cd build

cmake ..

make
```

Then the compiled application executable files are installed at the "bin" directory:

```
ls ./bin
```

# Partitioning

If we need to run the train data for the downstream tasks, such as Link prediction, the test data also should be processed.

```
cd build

./bin/mpgp -i [train_data] -e [test_data] -v [vertex_num] -p [partition_num] -t [float:0, integer:1]
```

The partitioned dataset will be saved in the input dataset directory. 

# Graph Embedding

To start the embedding, we fist need to cover the train graph to binary format

```
cd build

./bin/gconverter -i ../dataset/LJ.train-8-r -o ./LJ-8.data-r -s weighted
```

Then create the "out" directory to save the walks file or embedding file

```
mkdir out
```

### Run in Single-machine Environment
```
mpirun -np 8 ./bin/felog -g ../dataset/LJ-8.data-r -p ../dataset/LJ-8.part -v 2238731 -w 2238731 --min_L 20 --min_R 2 -o ./out/LJ --make-undirected -emb_output ./out/LJ_emb.txt -size 128 -iter 1 -threads 72 -window 10 -negative 5 -batch-size 21 -min-count 0 -sample 1e-3 -alpha 0.01 -cbow 0 -reuse-neg 0 -debug 2
```

### Run in Distributed Environment
- Copy the train dataset to the same path of each machine, or simply place it to a shared file system, such as NFS
- Touch a host file to write each machine's IP address, such as ./hosts
- Invoke the application with MPI 

```
mpirun -hostfile ./hosts -np 8 ./bin/felog -g ../dataset/LJ-8.data-r -p ../dataset/LJ-8.part -v 2238731 -w 2238731 --min_L 20 --min_R 2 -o ./out/LJ --make-undirected -emb_output ./out/LJ_emb.txt -size 128 -iter 1 -threads 72 -window 10 -negative 5 -batch-size 21 -min-count 0 -sample 1e-3 -alpha 0.01 -cbow 0 -reuse-neg 0 -debug 2
```

**Check the output files in "out" directory**

