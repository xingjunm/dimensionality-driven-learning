## Code for ICML 2018 paper "Dimensionality-Driven Learning with Noisy Labels".

## Update: Issues fixed on CIFAR-10. 11/07/2018
## Update: Cross entropy -> symmetric cross entropy at the begining on CIFAR-10. 10/23/2019

The Symmetric Cross Entropy was demonstrated can improve several exisiting methods including the D2L:
ICCV2019 "Symmetric Cross Entropy for Robust Learning with Noisy Labels"
https://arxiv.org/abs/1908.06112
https://github.com/YisenWang/symmetric_cross_entropy_for_noisy_labels

### 1. Train DNN models using command line:

An example: <br/>

```
python train_model.py -d mnist -m d2l -e 50 -b 128 -r 40 
```

`-d`: dataset in ['mnist', 'svhn', 'cifar-10', 'cifar-100'] <br/>
`-m`: model in ['ce', 'forward', 'backward', 'boot_hard', 'boot_soft', 'd2l'] <br/>
`-e`: epoch, `-b`: batch size, `-r`: noise rate in [0, 100] <br/> 


### 2. Run with pre-set parameters in main function of train_model.py:
```python
    for dataset in ['mnist']:
        for noise_ratio in ['0', '20', '40', '60']:
            args = parser.parse_args(['-d', dataset, '-m', 'd2l',
                                      '-e', '50', '-b', '128',
                                      '-r', noise_ratio])
            main(args)
```

#### Requirements:
tensorflow, Keras, numpy, scipy, sklearn, matplotlib
