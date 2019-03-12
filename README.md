#ADDA (Adversarial Discriminative Domain Adaptation)
tensorflow implement of ADDA [(paper)](https://arxiv.org/abs/1702.05464)

##Loss
####Classification loss with Ms (using Source Dataset)
```
tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=cls, labels=Y))
```
####Discriminator loss with Mt (using Source, Target Image)
log(D(Ms)) + log(1-D(Mt))
```
tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_source_logits, labels=tf.ones_like(D_source_logits)))
```
####Mt loss (using Target Image)
log(D(Mt))
```
tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_target_logits, labels=tf.zeros_like(D_target_logits)))
```

##How to Run
###Get Dataset
To generate MNIST-M dataset, first download [BSDS500](http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz) dataset to "_data/mnist_m/BSR_bsds500.tgz"
<br>
Then run create_mnistm.py which is from [github](https://github.com/pumpikano/tf-dann)
```
python create_mnistm.py
```
###How to Train
####First step
train Ms and Classifier with source dataset 
```
python main.py --train=source --learning_rate=0.001 --epoch=1000 --threshold=0.05 --batch_size=30
```
the ckpt files are saved as ./_models/source/source.ckpt and ./_models/classifier/classifier.ckpt 

Mt should be initializated with Ms. so copy the source.ckpt to target_pretrain folder and change the scope names.
```
cp ./_models/source/source.ckpt ./_models/target_pretrain/source.ckpt
cd ./_models/target_pretrain/source.ckpt
python source2target.py --checkpoint_dir=. --replace_from=source _cnn --replace_to=target_cnn
```

####Second step
train Mt and Discriminator with source and target image. 
```
python main.py --train=adda --learning_rate_D=0.001 --learning_rate_M=0.00001 --epoch=50 --threshold=0.96 --batch_size=30
```
the ckpt files are saved as ./_models/target/target.ckpt and ./_models/discriminater/discriminater.ckpt 

###How to Test
####Test Ms + Classifier with Target Dataset
```
python main.py --test=source
```
####Test Mt + Classifier with Target Dataset
```
python main.py --test=adda
```



