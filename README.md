# TART

This project a PyTorch implementation of  *Transition Matrix Representation of 
Trees with Transposed Convolutions*, published as a conference proceeding at
SDM 2022. The paper proposes TART (Transition Matrix Representation with
Transposed Convolutions), a novel framework for generalizing tree models with a
unifying view.

## Requirements

The repository sis written by Python 3.7 with the packages listed in
`requirements.txt`. A GPU environment is strongly recommended for efficient
training and inference of our model. You can type the following command to
install the required packages:
```
pip install -r requirements.txt
``` 

## Datasets

The paper uses 121 datasets from the UCI repository. Since the size of all
datasets is larger than 500 MB, we include only `balance-scale` in the current
repository, which is a sample dataset that includes 625 examples of 4 features
and 3 labels. You can download all datasets by running the following command:
```
bash uci.sh
```

## How to Run

You can test the performance of our TART by running a demo script:
```
bash demo.sh
```

The demo script runs the following Python command: 
```
python run.py 
    --gpu 0
    --workers 1
    --data balance-scale
    --model TART
    --depth 2
    --layers 1
    --style ensemble  
```

It runs for all datasets if no `data` argument is given. `workers` determines
the number of concurrent runs for each GPU. `style` is either `decision` or
`ensemble`, which represent the single- and multi-leaf selection scheme,
respectively. You can also provide other arguments such as the window size and
stride as described in the paper. Please refer to `main.py` for detailed
information, as `run.py` is just a wrapper script for efficient experiments.

## Reference

Please cite the following paper if you use the implementation.
```
@inproceedings{YooS22,
  title={Transition Matrix Representation of Trees with Transposed Convolutions},
  author={Jaemin Yoo and Lee Sael},
  booktitle={SIAM International Conference on Data Mining (SDM)},
  year={2022}
}
```
