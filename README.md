# symvae-aistats2024

Implementations of experiments from the paper (links follow).

The repository contains three directories: `mnist-ladder`, `mnist-reverse` and `fmnist-ladder`. Thereby, `-ladder` means the reverse encoder factorization order as in ladder-VAEs, whereas `-reverse` corresponds to the reverse encoder factorization order as in the Wake-Sleep algorithm, `mnist`/`fmnist` denote the dataset used.

The code is pretty similar in all three cases, hence, we shortly describe the usage right here, for the description of the learning methods we refer to the paper.

In order to learn models, go to the desired directory and type e.g.\
`export CUDA_VISIBLE_DEVICES=0; python3 main_sym.py --call_prefix 0 --nz0 30 --nz1 100 --stepsize 1e-4 --niterations 1000000` \
for Symmetric learning and \
`export CUDA_VISIBLE_DEVICES=2; python3 main_elbo.py --call_prefix 1 --nz0 30 --nz1 100 --stepsize 1e-4 --niterations 1000000` \
for learning by ELBO maximization. The above command-lines are examples, we used such calls ...
