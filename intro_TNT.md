$p(t^n|x) = \pi(t^n|x)*N(\delta x^n | v_x^n(x))*N(\delta y^n | v_y^n(x))$
$\delta x^n$和$\delta y^n$是GT对于anchor n的位移
$v_x^n(x)$和$v_y^n(x)$是模型输出的“对于anchor n的位移”的预测
$ \pi(t^n|x) = exp f(t^n, x) / \sum_{t'} exp f(t', x) $其实就是softmax，$f(t^n, x)$是模型输出的score
f()和v()是MLP
