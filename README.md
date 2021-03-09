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
----------
(The model file has been changed, can't run single node training so far.)
* without XLA
  * python script/train.py --mode=train  --model=DIEN

* with XLA
  * TF_XLA_FLAGS=--tf_xla_auto_jit=1 python script/train.py --mode=train  --model=DIEN


# Distributed Training (PS-Worker)
----------------------------------
Only support Async training now. Tested with NGC tf1-20.09 docker image.

* Cluster:
  * PS:10.149.160.49, Worker: 10.149.160.49, 10.149.160.63
  * (PS shall has a shared file system to store parameters, otherwise it should be in the same node with the 0 worker)

* Running commands:
  * PS$ CUDA_VISIBLE_DEVICES='' TF_CONFIG='{"cluster": {"worker": ["10.149.160.49:1112", "10.149.160.63:1113"],"ps": ["10.149.160.49:1111"]},"task": {"type": "ps", "index": 0}}' python script/train_dis.py --mode=train  --model=DIEN
  * Worker 0$ CUDA_VISIBLE_DEVICES='0' TF_CONFIG='{"cluster": {"worker": ["10.149.160.49:1112", "10.149.160.63:1113"],"ps": ["10.149.160.49:1111"]},"task": {"type": "worker", "index": 0}}' python script/train_dis.py --mode=train  --model=DIEN --batch_size=1024
  * Worker 1$ CUDA_VISIBLE_DEVICES='0' TF_CONFIG='{"cluster": {"worker": ["10.149.160.49:1112", "10.149.160.63:1113"],"ps": ["10.149.160.49:1111"]},"task": {"type": "worker", "index": 1}}' python script/train_dis.py --mode=train  --model=DIEN --batch_size=1024


* Verbs:
  * CUDA_VISIBLE_DEVICES='' RDMA_DEVICE=mlx5_0 TF_CONFIG='{"cluster": {"worker": ["10.10.10.1:1112"],"ps": ["10.10.10.2:1111"]},"task": {"type": "ps", "index": 0}, "rpc_layer": "grpc+verbs"}' python script/train_dis.py --mode=train  --model=DIEN --batch_size=1024

  * CUDA_VISIBLE_DEVICES='0' RDMA_DEVICE=mlx5_0 TF_CONFIG='{"cluster": {"worker": ["10.10.10.1:1112"],"ps": ["10.10.10.2:1111"]},"task": {"type": "worker", "index": 0}, "rpc_layer": "grpc+verbs"}' python script/train_dis.py --mode=train  --model=DIEN --batch_size=1024
