##ADDA (Adversarial Discriminative Domain Adaptation)
tensorflow implement of ADDA [(paper)](https://arxiv.org/abs/1702.05464)

###How to Train
To generate MNIST-M dataset, first download [BSDS500](http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz) dataset to "_data/mnist_m/BSR_bsds500.tgz"
<br>
Then run create_mnistm.py which is from [github](https://github.com/pumpikano/tf-dann)
<br>
```
python create_mnistm.py
```

```
python main.py --train --source --learning_rate=0.001 --epoch=1000 --threshold=0.05 --batch_size=30
python main.py --train --target --learning_rate_D=0.00 --epoch=50 --threshold=0.96 --batch_size=30
```

```
python main.py --test --source
python main.py --test --adda
```

###Model

###dataset

