This project aims to create a lightweight framework for DL distributed training based on TensorFlow and Pytorch.

# Two acceleration modes
It supports two distribution acceleration approaches:

1. Parameter Server (PS)
2. Collective communications

Take a look at [tips/core/ps](https://github.com/Superjomn/TiPS/tree/main/tips/core/ps) and [tips/core/collective](https://github.com/Superjomn/TiPS/tree/main/tips/core/collective) for more details.

# Current status
This is a part-time job when I am had a sick leave for fracture. The PS and Collective module themself are well developed and tested, while only the distributed training part is finished evaluation with real TensorFlow resnet50 model.

Currently, this project is hold for lack of time and further motivation.

# dependencies

- openmpi

Download from https://www.open-mpi.org/software/ompi/v4.1/

- ZeroMQ

apt-get install libzmq3-dev

# Usage
Run the following commands after compile the project.

``` sh
cd examples
mpirun --allow-run-as-root -np 4 python tensorflow2_keras_mnist.py
```

# References
I had read and learned the following projects:

1. [Horovod](https://github.com/horovod/horovod) for collective modules and TensorFlow and Pytorch support,
2. [SwiftSnails](https://github.com/Superjomn/SwiftSnails), my own project that implements a naive PS without MPI.
