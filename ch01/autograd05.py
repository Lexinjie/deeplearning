import matplotlib.pylab as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator
import numpy as np
import torch

def autograd():
    print('1.自动梯度计算')
    x = torch.arange(4.0, requires_grad=True)  # 1.将梯度附加到想要对其计算偏导数的变量
    print('x:', x)
    print('x.grad:', x.grad)
    y = 2 * torch.dot(x, x)  # 2.记录目标值的计算
    print('y:', y)
    y.backward()  # 3.执行它的反向传播函数
    print('x.grad:', x.grad)  # 4.访问得到的梯度，一个标量函数关于向量x的梯度是向量，并且与具有相同的形状。
    print('x.grad == 4*x:', x.grad == 4 * x)  #5.验证

    ## 计算另一个函数  在默认情况下，PyTorch会累积梯度，我们需要清除之前的值
    x.grad.zero_()
    y = x.sum()
    print('y:', y)
    y.backward()
    print('x.grad:', x.grad)

    # 非标量变量的反向传播
    print("*"*50)
    print("非标量变量的反向传播")
    x.grad.zero_()
    print('x:', x)
    y = x * x
    print("y:",y)
    y.sum().backward()
    print('x.grad:', x.grad)

    print('2.Python控制流的梯度计算')
    a = torch.tensor([2.0,1.0])  # 初始化变量
    a.requires_grad_(True)  # 1.将梯度赋给想要对其求偏导数的变量
    print('a:', a)
    d = f(a)  # 2.记录目标函数
    print('d:', d)
    #d.backward()  # <====== run time error if a is vector or matrix RuntimeError: grad can be implicitly created only for scalar outputs
    d.sum().backward()  # <===== this way it will work
    print('a.grad:', a.grad)  # 4.获取梯度


def f(a):
    b = a * 2
    print(b.norm())
    while b.norm() < 1000:  # 求L2范数：元素平方和的平方根
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c


def plotsinx():
    f,ax=plt.subplots(1)

    x = np.linspace(-3*np.pi, 3*np.pi, 100)
    x1= torch.tensor(x, requires_grad=True)
    y1= torch.sin(x1)
    y1.sum().backward()

    ax.plot(x,np.sin(x),label="sin(x)")
    ax.plot(x,x1.grad,label="gradient of sin(x)")
    ax.legend(loc="upper center", shadow=True)

    ax.xaxis.set_major_formatter(FuncFormatter(
    lambda val,pos: '{:.0g}\pi'.format(val/np.pi) if val !=0 else '0'
    ))
    ax.xaxis.set_major_locator(MultipleLocator(base=np.pi))

    plt.show()
