### Code for ICML 2018 paper "Dimensionality-Driven Learning with Noisy Labels".

#### - Update (2018.07): Issues fixed on CIFAR-10. 
#### - Update (2019.10): Start training with symmetric cross entropy (SCE) loss (replacing cross entropy).

The Symmetric Cross Entropy (SCE) was demonstrated can improve several exisiting methods including the D2L:
ICCV2019 "Symmetric Cross Entropy for Robust Learning with Noisy Labels"
https://arxiv.org/abs/1908.06112
https://github.com/YisenWang/symmetric_cross_entropy_for_noisy_labels

#### - Update (2020.03): convergence issue on CIFAR-100 when using SCE loss: learning rate, data augmentation and parameters for SCE. 


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
    # mnist example
    args = parser.parse_args(['-d', 'mnist', '-m', 'd2l',
                              '-e', '50', '-b', '128',
                              '-r', '40'])
    main(args)
    
    # svhn example
    args = parser.parse_args(['-d', 'svhn', '-m', 'd2l',
                              '-e', '50', '-b', '128',
                              '-r', '40'])
    main(args)
    
    # cifar-10 example
    args = parser.parse_args(['-d', 'cifar-10', '-m', 'd2l',
                              '-e', '120', '-b', '128',
                              '-r', '40'])
    main(args)
    
    # cifar-100 example
    args = parser.parse_args(['-d', 'cifar-100', '-m', 'd2l',
                              '-e', '200', '-b', '128',
                              '-r', '40'])
    main(args)
```

#### Requirements:
tensorflow, Keras, numpy, scipy, sklearn, matplotlib
