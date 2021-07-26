# bayesian-dependency-parsing

Trying to implement a Bayesian neural network to perform graph-based semantic dependency parsing! General problem setup and inspiration  is from [Simpler but More Accurate Semantic Dependency Parsing](https://arxiv.org/abs/1807.01396)

## setup

Required libraries:
```
tensorflow - 1.15
tensorflow probability - 1.15
[edward](https://github.com/blei-lab/edward)- 1.3.5
```

## running

Edit the `Config` class in the respective file and run `python {classifier/bayesian_classifier}.py.` Training and inference should take about 30 minutes without a GPU, within 5 when using one.
Results:

[Short paper](https://docs.google.com/viewer?url=https://raw.githubusercontent.com/TarunSunkaraneni/bayesian-dependency-parsing/master/report.pdf)

![results](/results.png?raw=true "results") 