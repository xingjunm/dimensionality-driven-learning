## Code for ICML 2018 paper "Dimensionality-Driven Learning with Noisy Labels".

## Updated: change initial learning rates to 0.01 for convergence issue. This leads to performance decreases, still needs to be fixed.
## Updated: uploaded old version of d2l. 20/09/2018

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
tensorflow, Keras, numpy, scipy, tqdm, sklearn, matplotlib
