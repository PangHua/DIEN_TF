# Reference

* https://arxiv.org/pdf/1809.03672.pdf
* https://github.com/alibaba/ai-matrix/tree/master/macro_benchmark/DIEN


# Dataset

Download the processed data from https://github.com/mouna99/dien

* tar -jxvf data.tar.gz
* mv data/* .
* tar -jxvf data1.tar.gz
* mv data1/* .
* tar -jxvf data2.tar.gz
* mv data2/* .

When you see the files below, you can do the next work.

* cat_voc.pkl
* mid_voc.pkl
* uid_voc.pkl
* local_train_splitByUser
* local_test_splitByUser
* reviews-info
* item-info


# Training
--------
* without XLA
  * python script/train.py --mode=train  --model=DIEN

* with XLA
  * TF_XLA_FLAGS=--tf_xla_auto_jit=1 python script/train.py --mode=train  --model=DIEN
