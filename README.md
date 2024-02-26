

## Running FPL on benchmark datasets (MNIST,SVHN, CIFAR-10 and CIFAR-100)
Here is an example: 

```bash
python FPL.py --dataset cifar10 --noise_type symmetric --noise_rate 0.2 --alpha 0.9 --beta 0.09 --gamma 0.01
```

```bash
python FPL.py --dataset cifar10 --noise_type instance --noise_rate 0.2 --alpha 0.9 --beta 0.099 --gamma 0.001
```

```bash
python FPL.py --dataset cifar100 --noise_type symmetric --noise_rate 0.2 --alpha 0.9 --beta 0.099 --gamma 0.001
```

```bash
python FPL.py --dataset mnist --noise_type symmetric --noise_rate 0.2 --alpha 0.9 --beta 0.09 --gamma 0.01
```

```bash
python FPL.py --dataset SVHN --noise_type symmetric --noise_rate 0.2 --alpha 0.9 --beta 0.099 --gamma 0.001
```

## Running FPL on one real dataset (Clothing1M)
Here is an example with lambda setting: 

```bash
python FPL_clothing.py --dataset clothing1M --alpha 0.9 --beta 0.099 --gamma 0.001
```

## Running FPL on imblance datasets (im-MNIST,im-SVHN)
Here is an example with lambda setting: 

```bash
python FPL_imblance.py --dataset mnist --alpha 0.9 --beta 0.09 --gamma 0.01
```

```bash
python FPL_imblance.py --dataset SVHN --alpha 0.9 --beta 0.09 --gamma 0.01
```


