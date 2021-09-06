# bayesian-dependency-parsing

Trying to implement a Bayesian neural network to perform graph-based semantic dependency parsing! General problem setup and inspiration  is from [A Fast and Accurate Dependency Parser using Neural Networks](https://aclanthology.org/D14-1082.pdf)

## setup

Required libraries:
```
tensorflow - 1.15
tensorflow probability - 1.15
```

## running

Edit the `Config` class in the respective file and run `python {classifier/bayesian_classifier}.py.` Training and inference should take about 30 minutes without a GPU, within 5 when using one.

## results:

![results](/results.png?raw=true "results") 


[Short paper](https://docs.google.com/viewer?url=https://raw.githubusercontent.com/TarunSunkaraneni/bayesian-dependency-parsing/master/report.pdf)
