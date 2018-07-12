# Tensorflow study project

## Reference
  - TensorFlow - Getting Started
  - https://github.com/akanshajainn/Machine-Learning---Google-Developers
  - Building Machine Learning Systems with TensorFlow 
  - http://web.stanford.edu/class/cs20si/
    - http://web.stanford.edu/class/cs20si/lectures/
    - https://github.com/chiphuyen/stanford-tensorflow-tutorials


## Tensfor flow setup
  Using a common TensorFLow env for all projects
```  
  $mkdir ~/tensorflow
  $ virtualenv --system-site-packages ~/tensorflow              # for Python 2.7
  $ virtualenv --system-site-packages -p python3 ~/tensorflow   # for Python 3.n

  $ source ~/tensorflow/bin/activate
  (tensorflow)$ easy_install -U pip
  (tensorflow)$ pip install --upgrade tensorflow      # for Python 2.7
  (tensorflow)$ pip3 install --upgrade tensorflow     # for Python 3.n
```

## Setup python
```
$ virtualenv --system-site-packages -p python3 venv3

$ . venv3/bin/activate

(venv3)$ pip3 install -r requirements.txt

...
(venv3)$ deactivate
```

## Tensor board
```
  tensorboard --logdir=. --port=8000
```