### About
It is an attempt to reproduce results of the original project. For the original Readme see the original `readme.txt`.

#### py2brian
There I started to refactor the existing brian1 version but it is a very time consuming process as the existing code is
sufficiently tangled and brian1 can't be run properly on my machine due to `pylab/scipy`. Old versions of them have to
be installed. On my machine those old versions are not installable. That is why I freeze the work there and start clean
in Brian2 and py3.
Also I am not sure that the original can reproduce the claimed results. Too high probability of an error in this tangled
code. For example, I'm highly curios why weights of `XeAe` are not updated when I was able to run the original without
plotting functions.

#### py3brian2
Basically we just take an idea and try to build the same in the latest Brian2.

#### Existing rewrites of the original
- https://github.com/djsaunde/lm-snn
- https://github.com/zxzhijia/Brian2STDPMNIST
- https://github.com/whenov/stdp-mnist-brian2
