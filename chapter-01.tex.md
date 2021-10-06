## 神经网络的复习
- src/ch01/forward_net.py 简单的两层前向传播网络
- src/ch01/show_spiral_dataset.py 展示spiral形状的数据分布
- src/ch01/train.py 训练次数和loss函数值的关系图
- src/ch01/two_layer_net.py 两层网络封装类，含前向和反向传播
---
决策边界：学习后的神经网络的区域划分。

![](./images/visualize-decision-boundary.png)

> src/ch01/train_custom_loop.py 捕获漩涡模式

说明学习后的神经网络可以正确地捕获“旋涡”这个模式。
也就说，模型正确地学习了非线性的区域划分。

### 计算的高速化
#### 位精度
默认是float64
```python
>>> import numpy as np
>>> a = np.random.randn(3)
>>> a.dtype
dtype('float64')
```
指定float32
```python
>>> b = np.random.randn(3).astype(np.float32)
>>> b.dtype
dtype('float32')
```
将权重数据用 16 位精度保存时，只需要 32 位时的一半容量。
因此，仅在保存学习好的权重时，将其变换为 16 位浮点数。
> Google TPU support 8 bit computation。

#### GPU（CuPy）
先安装CuDA：
> https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html

然后验证安装, 得知Cuda版本为 11.1
```
$ nvcc -V
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2020 NVIDIA Corporation
Built on Mon_Oct_12_20:54:10_Pacific_Daylight_Time_2020
Cuda compilation tools, release 11.1, V11.1.105
Build cuda_11.1.relgpu_drvr455TC455_06.29190527_0
```
接下来，安装对应版本的cupy，注意版本必须对应: 
- https://docs.cupy.dev/en/latest/install.html

```bash
pip install cupy-cuda111
```

测试cupy
```bash
$ python
Python 3.7.4 (tags/v3.7.4:e09359112e, Jul  8 2019, 20:34:20) [MSC v.1916 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license" for more information.
>>> import cupy as cp
>>> x = cp.arange(6).reshape(2, 3).astype('f')
>>> x
array([[0., 1., 2.],
       [3., 4., 5.]], dtype=float32)
```