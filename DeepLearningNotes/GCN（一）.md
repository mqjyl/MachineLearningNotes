---
title: GCN（一）
tags: 卷积,CNN,GCN,深度学习,神经网络
grammar_cjkRuby: true
grammar_html: true
---

#### 一、GCN（Graph Convalutional Network）
###### （一）、为什么要研究GCN?
　　CNN处理的图像或视频数据中的像素点是排列成很整齐的矩阵，也就是很多论文中提到的（Euclidean Structure），而很多时候我们需要处理的是Non Euclidean Structure的数据，比如社交网络。
  
![enter description here](./images/5.jpg)

　　这种网络结构在图论中被抽象成图谱图。
  １、CNN无法处理Non Euclidean Structure的数据，传统的离散卷积在Non Euclidean Structure的数据上无法保持平移不变形，因为拓扑图中每个顶点的相邻顶点数目都可能不同，无法用同样的卷积核来进行卷积运算。
  ２、GCN的研究重点在于在拓扑图（Non Euclidean Structure）上有效地提取空间特征来进行机器学习。
###### （二）、提取拓扑图中空间特征的两种方式
１、**Vertex domain(spatial domain)**：本质是提取拓扑图上每个顶点相邻的neighbors。
解决两个问题：
　　(1)、根据什么条件去找中心vertex的neighbors，也就是如何确定receptive field？
　　(2)、给定了receptive field，按照什么方式处理包含不同数目neighbours的特征？
	   这种方式很明显要对单个顶点做计算处理，提取的图的空间特征相当于对每个顶点的处理结果的集合。
２、**Spectral domain**：这是GCN的理论基础，本质是借助图论的相关理论来实现拓扑图上的卷积操作。
理论基础：
　　(1)、谱图理论(Spectral Graph Theory)：借助图的拉普拉斯矩阵的特征值和特征向量研究图的性质。
　　(2)、Graph上的傅里叶变换(Fourier Transformation)
　　(3)、GSP(graph signal processing)
  
#### 二、 图的拉普拉斯矩阵
###### （一）、对于图 `!$G=(V,E)$` ，常见的拉普拉斯矩阵有三种：
**No.1 Combinatorial Laplacian(组合拉普拉斯)：`!$L=D-A$`**
其中 `!$L$` 是Laplacian矩阵，`!$D$` 是顶点的度矩阵，是一个对角矩阵，对角上的元素依次为各个顶点的度，`!$A$` 是图的邻接矩阵，计算方法示例如图：

![enter description here](./images/6.jpg)

其中 `!$L$` 由下列公式给出：
```mathjax!
$$
L_{i,j}:=
\begin{cases}
deg(v_i) &  & {if \ i = j}                                  \\
-1       &  & {if\ i\neq j\ and\ v_i\ is\ adjacent\ to\ v_j} \\
0        &  & {otherwise}
\end{cases}
$$
```
其中`!$deg(v_i)$` 是顶点`!$i$` 的度。
 **No.2 Symmetric normalized Laplacian(对称归一化拉普拉斯)：`!$L^{sys}:=D^{-1/2}LD^{-1/2}=I-D^{-1/2}AD^{-1/2}$`**
	   其中 `!$L_{i,j}^{sym}$` 由下列公式给出：
```mathjax!
$$
L_{i,j}^{sym}:=
\begin{cases}
1        &  & {if \ i = j}\ and\ deg(v_i)\neq 0                 \\
-\frac{1}{\sqrt{deg(v_i)deg(v_j)}} & & {if\ i\neq j\ and\ v_i\ is\ adjacent\ to\ v_j} \\
0        &  & {otherwise}
\end{cases}
$$
```
我们可以看看 `!$D^{-1/2}AD^{-1/2}$`发生了什么，`!$D$` 为只有主对角线上元素非０的对角阵，`!$A$`中记录顶点间的邻接信息，  `!$D^{-1/2}AD^{-1/2}$`　使得`!$A$`中第`!$i$`行和第`!$i$`列的值都除以`!$\sqrt{D_{ii}}$`。
 **No.3 Random walk normalized Laplacian(随机游走归一化拉普拉斯)：`!$L^{rw}:=D^{-1}L=I-D^{-1}A$`**
	   其中 `!$L_{i,j}^{rw}$` 可以由下列方式计算：
```mathjax!
$$
L_{i,j}^{sym}:=
\begin{cases}
1        &  & {if \ i = j}\ and\ deg(v_i)\neq 0                 \\
-\frac{1}{deg(v_i)} & & {if\ i\neq j\ and\ v_i\ is\ adjacent\ to\ v_j} \\
0        &  & {otherwise}
\end{cases}
$$
```
###### （二）、对于拉普拉斯矩阵定义的理解
　　１、先从拉普拉斯算子说起，拉普拉斯算子数学定义是这样的： 
```mathjax!
$$\triangle = \sum_i\frac {\partial^2} {\partial x_i^2}\\$$
```
其含义很明确，是非混合二阶偏导数的和！
　　２、再看图像处理上是怎么近似的：
　　图像是一种离散数据，那么其拉普拉斯算子必然要进行离散化。由导数定义：
```mathjax!
$$
\begin{aligned} f'(x) &= \frac {\partial f(x)}{\partial x}\\ & = \lim_{\delta \to 0} \frac{f(x+\delta)-f(x)}{\delta}\\ & \approx^{离散化}
f(x+１)-f(x)\end{aligned}\\
$$
```
　　得出：
```mathjax!
$$
\begin{aligned} \frac {\delta^2 f(x)}{\delta x^2} &= f''(x) \\ &\approx f'(x)-f'(x-1) \\ &\approx f(x+1)-f(x) - (f(x) - f(x-1))\\ &=f(x+1)+f(x-1)-2f(x) \end{aligned}\\ 
$$
```
**结论1：二阶导数近似等于其二阶差分。
结论2：二阶导数等于其在所有自由度上微扰之后获得的增益。**
　　一维函数其自由度可以理解为 `!$2$`，分别是 `!$+1$` 和 `!$-1$` 两个方向。对于二维的图像来说，其有两个方向（`!$4$` 个自由度）可以变化，即如果对 `!$(x,y)$` 处的像素进行扰动，其可以变为四种状态 `!$(x+1,y)，(x-1,y)，(x,y+1)，(x,y-1)$`。当然了，如果将对角线方向也认为是一个自由度的话，会再增加几种状态 `!$(x+1,y+1)，(x+1,y-1)，(x-1,y+1)，(x-1,y-1)$`，事实上图像处理正是这种原理。
　　同理，将拉普拉斯算子离散化：
```mathjax!
$$
\begin{aligned} \triangle &=\frac {\delta^2 f(x,y)}{\delta x^2} + \frac {\delta^2 f(x,y)}{\delta y^2} \\ &\approx f(x+1,y)+f(x-1,y)-2f(x,y) + [f(x,y+1)+f(x,y-1)-2f(x,y)]\\ &= f(x+1,y)+f(x-1,y)+f(x,y+1)+f(x,y-1)-4f(x,y) \end{aligned}\\
$$
```
　　上式可以理解为，在图像上某一点，其拉普拉斯算子的值，即为对其进行扰动，使其变化到相邻像素后得到的增益。这给我们一种形象的结论：拉普拉斯算子就是在所有自由度上进行微小变化后获得的增益。
　　推广到Graph，对于有 `!$N$` 个节点的Graph，就设节点为 `!$1,...,N$` 吧，且其邻接矩阵为 `!$A$`。这个Graph的自由度最多为 `!$N$`。因为如果该图是一个完全图，即任意两个节点之间都有一条边，那么对一个节点进行微扰，它可能变成任意一个节点。那么上面的函数 `!$f$` 就理所当然是一个 `!$N$` 维的向量，即：
```mathjax!
$$f = (f_1,...,f_N)$$
```
　　其中 `!$f_i$` 即表示函数 `!$f$` 在节点 `!$i$` 的值。类比 `!$f(x,y)$` 即为 `!$f$` 在 `!$(x,y)$` 处的值。对于任意节点 `!$i$` 进行微扰，它可能变为任意一个与他相邻的节点 `!$j\in\mathcal{N_i}$` ，其中 `!$\mathcal{N_i}$` 表示节点 `!$i$` 的一阶邻域节点。
　　对于Graph，从节点 `!$i$` 变化到节点 `!$j$` 增益是多少呢？即 `!$f_j-f_i$` 是多少？最容易想到就是和他们之间的边权相关。此处用 `!$A_{ij}$` 表示。那么，对于节点 `!$i$` 来说，其变化的增益:
```mathjax!
$$\sum_{j\in\mathcal{N}_i}A_{ij}[f_j-f_i]$$
```
　　所以，对于Graph来说，其拉普拉斯算子如下：
```mathjax!
$$
\begin{aligned} (\triangle f)_i &=\sum_i \frac {\delta^2 f} {\delta i^2}\\ &\approx \sum_{j\in\mathcal{N}_i}A_{ij}[f_j-f_i] \end{aligned}\\
$$
```
　　上式 `!$j\in\mathcal{N}_i$` 可以去掉，因为节点 `!$i$` 和 `!$j$` 不直接相邻的话， `!$A_{ij} = 0$`；
　　继续化简一下：
```mathjax!
$$
\begin{aligned} \sum_{j\in\mathcal{N}_i}A_{ij}[f_j-f_i]&=\sum_{j}A_{ij}f_j - \sum_{j}A_{ij}f_i\\ &=(Af)_i - (Df)_i \\ &=[(A-D)f]_i \end{aligned}\\
$$
```
　　即:
```mathjax!
$$(\triangle f)_i =[(A-D)f]_i$$
```
　　对于任意的 `!$i$` 成立，那么也就是：
```mathjax!
$$\triangle f \equiv(A-D)f $$
```
　　因此图上的拉普拉斯算子应该定义为 `!$A-D$`。
###### （三）、拉普拉斯矩阵的性质
１、拉普拉斯矩阵是对称矩阵，可以进行特征分解（谱分解），这就是GCN的spectral domain所依据的。
２、拉普拉斯矩阵只在中心顶点和一阶相连的顶点上（1-hop neighbor）有非0元素，其余之处均为0。
３、拉普拉斯矩阵存在一个为零的特征值（秩为|V|-1），其余特征值大于零，因此为半正定矩阵。
###### （四）、拉普拉斯矩阵的谱分解
　　首先说明的是，矩阵的特征分解、谱分解、对角化是同一个概念。
　　特征分解（Eigendecomposition），又称谱分解（Spectral decomposition）是将矩阵分解为由其特征值和特征向量表示的矩阵之积的方法。矩阵可以特征分解的充要条件为n阶方阵存在n个线性无关的特征向量，即可对角化。但是拉普拉斯矩阵是半正定对称矩阵（半正定矩阵本身就是对称矩阵），有如下三个性质：
 - 对称矩阵一定n个线性无关的特征向量
 - 半正定矩阵的特征值一定非负
 - 对阵矩阵的特征向量相互正交，即所有特征向量构成的矩阵为正交矩阵。

由上可以知道拉普拉斯矩阵一定可以谱分解，且分解后有特殊的形式。
对于拉普拉斯矩阵，起谱分解为：
```mathjax!
    $$
	L=U\begin{pmatrix}
    \lambda_1 & & \\
                         &\ddots & \\
                         & &\lambda_n \\
    \end{pmatrix}U^{-1}
	$$
```
其中 `!$U=(\vec{u_1},\vec{u_2},\cdots,\vec{u_n}) $`是列向量为单位特征向量的矩阵，也就说 `!$\vec{u_l}$` 是列向量。
```mathjax!
    $$
	\begin{pmatrix}
    \lambda_1 & & \\
                         &\ddots & \\
                         & &\lambda_n \\
    \end{pmatrix}
	$$
```
是 n 个特征值构成的对角阵。
由于 `!$U$` 是正交矩阵，即 `!$UU^{T}=E$`， 所以特征分解又可以写成：
```mathjax!
    $$
	L=U\begin{pmatrix}
    \lambda_1 & & \\
                         &\ddots & \\
                         & &\lambda_n \\
    \end{pmatrix}U^{T}
	$$
```
注意的是特征分解最右边的是特征矩阵的逆，只是拉普拉斯矩阵的性质才可以写成特征矩阵的转置。

#### 三、 Graph上的傅里叶变换
###### （一）、傅里叶变换
 ```mathjax!
 $$
 F(\omega)=\mathcal{F}[f(t)]=\int_{}^{}f(t)e^{-i\omega t} dt
 $$
 ```
传统的傅里叶变换定义为信号 `!$f(t)$` 与基函数 `!$e^{-i\omega t}$` 的积分，那么为什么要找 `!$e^{-i\omega t}$` 作为基函数呢？从数学上看， `!$e^{-i\omega t}$` 是拉普拉斯算子的特征函数（满足特征方程），`!$ \omega $` 就和特征值有关。
 广义的特征方程定义为：
```mathjax!
$$A V=\lambda V$$
```
其中 A 是一种变换， V 是特征向量或者特征函数（无穷维的向量），`!$ \lambda $` 是特征值。
`!$e^{-i\omega t} $` 满足：
```mathjax!
$$
 \Delta e^{-i\omega t}=\frac{\partial^{2}}{\partial t^{2}} e^{-i\omega t}=-\omega^{2} e^{-i\omega t}\
 $$
 ```
当然 `!$e^{-i\omega t} $` 就是变换 `!$\Delta$` 的特征函数， `!$\omega$` 和特征值密切相关。
在处理Graph问题的时候，用到拉普拉斯矩阵（拉普拉斯矩阵就是离散拉普拉斯算子），对其进行特征分解，`!$L$` 是拉普拉斯矩阵， `!$V$` 是其特征向量，自然满足特征方程：
```mathjax!
$$LV=\lambda V $$
```
离散积分就是一种内积形式，仿上定义Graph上的傅里叶变换：
```mathjax!
$$
F(\lambda_l)=\hat{f}(\lambda_l)=\sum_{i=1}^{N}{f(i) u_l^*(i)}
$$
```
`!$f$` 是Graph上的 `!$N$` 维向量，`!$f(i)$` 与Graph的顶点一一对应， `!$u_l(i)$` 表示第 `!$l$` 个特征向量的第 `!$i$` 个分量。那么特征值（频率） `!$\lambda_l$` 下的，`!$f$` 的Graph傅里叶变换就是与 `!$\lambda_l$` 对应的特征向量 `!$u_l(i)$` 进行内积运算。
注：上述的内积运算是在复数空间中定义的，所以采用了`!$u_l^*$`，也就是特征向量 `!$u_l$` 的共轭。
利用矩阵乘法将Graph上的傅里叶变换推广到矩阵形式：
```mathjax!
$$
\left(\begin{matrix} \hat{f}(\lambda_1)\\ \hat{f}(\lambda_2) \\ \vdots \\\hat{f}(\lambda_N) \end{matrix}\right)=\left(\begin{matrix}\ u_1(1) &u_1(2)& \dots &u_1(N) \\u_2(1) &u_2(2)& \dots &u_2(N)\\ \vdots &\vdots &\ddots & \vdots\\ u_N(1) &u_N(2)& \dots &u_N(N) \end{matrix}\right)\left(\begin{matrix}f(1)\\ f(2) \\ \vdots \\f(N) \end{matrix}\right)
$$
```
即 `!$f$` 在Graph上傅里叶变换的矩阵形式为：`!$\hat{f}=U^{-1}f \qquad(a)$`， 式中 `!$U^{-1}$` 等于 `!$U^T$`。
###### （二）、傅里叶逆变换
传统的傅里叶逆变换是对频率 `!$\omega $` 求积分：
```mathjax!
$$
\mathcal{F}^{-1}[F(\omega)]=\frac{1}{2\Pi}\int_{}^{}F(\omega)e^{i\omega t} d\omega
$$
```
迁移到Graph上变为对特征值 `!$\lambda_l$` 求和：
```mathjax!
$$f(i)=\sum_{l=1}^{N}{\hat{f}(\lambda_l) u_l(i)}$$
```
利用矩阵乘法将Graph上的傅里叶逆变换推广到矩阵形式：
```mathjax!
$$
\left(\begin{matrix}f(1)\\ f(2) \\ \vdots \\f(N) \end{matrix}\right)= \left(\begin{matrix}\ u_1(1) &u_2(1)& \dots &u_N(1) \\u_1(2) &u_2(2)& \dots &u_N(2)\\ \vdots &\vdots &\ddots & \vdots\\ u_1(N) &u_2(N)& \dots &u_N(N) \end{matrix}\right) \left(\begin{matrix} \hat{f}(\lambda_1)\\ \hat{f}(\lambda_2) \\ \vdots \\\hat{f}(\lambda_N) \end{matrix}\right)
$$
```
即 `!$f$` 在Graph上傅里叶逆变换的矩阵形式为：`!$f=U\hat{f} \qquad(b)$`。

#### 四、 Graph上的卷积
　　卷积定理：函数卷积的傅里叶变换是函数傅立叶变换的乘积，即对于函数 `!$f(t)$` 与 `!$h(t)$` 两者的卷积是其函数傅立叶变换乘积的逆变换：
  ```mathjax!
  $$
f*h=\mathcal{F}^{-1}\left[ \hat{f}(\omega)\hat{h}(\omega) \right]=\frac{1}{2\Pi}\int_{}^{} \hat{f}(\omega)\hat{h}(\omega)e^{i\omega t} d\omega
$$
```
利用卷积定理类比来将卷积运算，推广到Graph上，并把傅里叶变换的定义带入，`!$f$` 与卷积核 `!$h$` 在Graph上的卷积可按下列步骤求出：
`!$f$` 的傅里叶变换为 `!$\hat{f}=U^Tf$`，卷积核 `!$h$` 的傅里叶变换写成对角矩阵的形式即为：
```mathjax!
$$
\left(\begin{matrix}\hat h(\lambda_1) & \\&\ddots \\ &&\hat h(\lambda_n) \end{matrix}\right)
$$
```
`!$\hat{h}(\lambda_l)=\sum_{i=1}^{N}{h(i) u_l^*(i)}$` 是根据需要设计的卷积核 `!$h$`在Graph上的傅里叶变换。
两者的傅立叶变换乘积即为：
```mathjax!
$$
\left(\begin{matrix}\hat h(\lambda_1) & \\&\ddots \\ &&\hat h(\lambda_n) \end{matrix}\right)U^Tf
$$
```
再乘以 `!$U$` 求两者傅立叶变换乘积的逆变换，则求出卷积：
```mathjax!
$$
(f*h)_G= U\left(\begin{matrix}\hat h(\lambda_1) & \\&\ddots \\ &&\hat h(\lambda_n) \end{matrix}\right) U^Tf \qquad(1)
$$
```
式中： `!$U$`及 `!$U^{T}$` 的定义与前面的相同。
注：很多论文中的Graph卷积公式为：
```mathjax!
$$(f*h)_G=U((U^Th)\odot(U^Tf)) \qquad(2)$$
```
`!$\odot $` 表示hadamard product（哈达马积），对于两个向量，就是进行内积运算；对于维度相同的两个矩阵，就是对应元素的乘积运算。
其实式(2)与式(1)是完全相同的。因为
```mathjax!
$$\left(\begin{matrix}\hat h(\lambda_1) & \\&\ddots \\ &&\hat h(\lambda_n) \end{matrix}\right)$$
```
与 `!$U^Th$` 都是 `!$h$` 在Graph上的傅里叶变换。
而根据矩阵乘法的运算规则：对角矩阵
```mathjax!
$$\left(\begin{matrix}\hat h(\lambda_1) & \\&\ddots \\ &&\hat h(\lambda_n) \end{matrix}\right)$$
```
与 `!$U^Th$` 的乘积和 `!$U^Th$` 与 `!$U^Tf$` 进行对应元素的乘积运算是完全相同的。

#### 五、 为什么拉普拉斯矩阵的特征向量可以作为傅里叶变换的基？特征值表示频率？
###### (1)为什么拉普拉斯矩阵的特征向量可以作为傅里叶变换的基？
傅里叶变换一个本质理解就是：把任意一个函数表示成了若干个**正交函数**（由sin,cos 构成）的线性组合。

![enter description here](./images/7.jpg)

通过第5节中(b)式可以看出，graph傅里叶变换把graph上定义的任意向量 `!$f$` ，表示成了拉普拉斯矩阵特征向量的线性组合，即：
```mathjax!
$$
f=\hat{f}(\lambda_1)u_1+\hat{f}(\lambda_2)u_2+\cdots +\hat{f}(\lambda_n)u_n
$$
```
那么：为什么graph上任意的向量 `!$f$` 都可以表示成这样的线性组合？
原因在于 `!$(\vec{u_1},\vec{u_2},\cdots,\vec{u_n})$` 是graph上 `!$n$` 维空间中的 `!$n$` 个线性无关的正交向量，由线性代数的知识可以知道： `!$n$` 维空间中  `!$n$` 个线性无关的向量可以构成空间的一组基，而且拉普拉斯矩阵的特征向量是一组正交基。
###### (2)怎么理解拉普拉斯矩阵的特征值表示频率？
将拉普拉斯矩阵 `!$L$` （半正定矩阵）的 `!$n$` 个非负实特征值，从小到大排列为 `!$\lambda_1 \le \lambda_2 \le \cdots \le \lambda_n$`，而且最小的特征值 `!$\lambda_1=0$`，因为 `!$n$` 维的全 1 特征**strong text**向量对应的特征值为 0（由 L 的定义就可以得出）：
```mathjax!
$$L \left(\begin{matrix}1\\ 1 \\ \vdots \\1 \end{matrix}\right)=0$$
```
从特征方程
```mathjax!
$$Lu=\lambda u $$
```
的数学理解来看，在由Graph确定的 `!$n$` 维空间中，越小的特征值 `!$\lambda_l $` 表明：拉普拉斯矩阵 `!$L$` 其所对应的基 `!$u_l $`上的分量、“信息”越少，当然就是可以忽略的低频部分。其实图像压缩就是这个原理，把像素矩阵特征分解后，把小的特征值（低频部分）全部变成 0，PCA（principal component analysis ( 主成分分析)）降维也是同样的，把协方差矩阵特征分解后，按从大到小取出前 K 个特征值对应的特征向量作为新的“坐标轴”。

#### 六、 GCN(Graph Convolutional Network)
　　Deep learning 中的 Convolution 就是要设计含有 trainable 共享参数的kernel，从(1)式看很直观：graph convolution 中的卷积参数就是 `!$diag(\hat h(\lambda_l) )$`（对角阵）。
###### 第一代的GCN(Spectral Networks and Deep Locally Connected Networks on Graph)简单粗暴地把 `!$diag(\hat h(\lambda_l) )$` 变成了卷积核 `!$diag(\theta_l )$`，也就是：
```mathjax!
$$
y_{output}=\sigma \left(U\left(\begin{matrix}\theta_1 &\\&\ddots \\ &&\theta_n \end{matrix}\right) U^T x \right) \qquad(3)
$$
```
　　式（3）就是标准的第一代GCN中的layer了，其中 `!$\sigma(\cdot)$` 是激活函数，`!$\Theta=({\theta_1},{\theta_2},\cdots,{\theta_n})$` 就跟三层神经网络中的 weight 一样是任意的参数，通过初始化赋值然后利用误差反向传播进行调整， x 就是graph上对应于每个顶点的 feature vector（由特数据集提取特征构成的向量）。
　　1）这里的spectral graph convolution指的是
```mathjax!
$$
X_{t+1}=U\ diag(\theta_i)\ U^T X_t  \qquad(3)
$$
```
其中 `!$X$` 表示每个节点上的特征，`!$U$` 是该graph的Laplacian矩阵的特征向量，`!$diag(\theta_i)$` 是一个对角阵，也可以看做是对特征值 `!$\Lambda$` 的调整 。
　　2）为什么这里可以对应于卷积：`!$U^TX$` 将节点投影到了频率域，`!$drag(\theta_i)\ U^TX$`表示频率域的乘积，频率域乘积对应于空间域的卷积。最后再乘以 `!$U$` 将频率域变换回到空间域。
　　3）为什么 `!$U^TX$` 就是频率域变换：我们可以这么理解： Laplacian 矩阵的定义是 `!$L=D-A$`。把 `!$L$` 的特征值按从小到大排列，那么对应的特征向量，正好对应于图上的从低频到高频基。 就像我们对一维函数做傅里叶变换，就是将不同频率的基( `!$sin(m\theta)cos(m\theta$`) )与该函数进行內积（这里就是积分）一样，这里我们也对图上每个节点的值与该图的不同频率的基（特征向量）进行內积。
　　**弊端：**
　　１、每一次前向传播，都要计算 `!$U$`，`!$diag(\theta_l )$` 及 `!$U^T$` 三者的乘积，特别是对于大规模的graph，计算的代价较高， 计算复杂度为`!$\mathcal{O}(n^2)$`。
　　２、卷积核需要 n 个参数。相当于每个卷积核都跟图像一样大。
　　３、卷积核的spatial localization不好，这是相对第二代卷积核而言的。
###### 第二代的GCN(Convolutional Neural Networks on Graphs With Fast Localized Spectral Filtering)把 `!$\hat h(\lambda_l)$` 巧妙地设计成了 `!$\sum_{j=0}^K \alpha_j \lambda^j_l$` ，也就是：
```mathjax!
$$
 y_{output}=\sigma \left(U\left(\begin{matrix}\sum_{j=0}^K \alpha_j \lambda^j_1 &\\&\ddots \\ && \sum_{j=0}^K \alpha_j \lambda^j_n \end{matrix}\right) U^T x \right) \qquad(4)
$$
```
　　利用矩阵乘法进行变换：
```mathjax!
$$
 \left(\begin{matrix}\sum_{j=0}^K \alpha_j \lambda^j_1 &\\&\ddots \\ && \sum_{j=0}^K \alpha_j \lambda^j_n \end{matrix}\right)=\sum_{j=0}^K \alpha_j \Lambda^j
$$
```
　　因为 `!$L^2=U \Lambda U^TU \Lambda U^T=U \Lambda^2 U^T $` 且 `!$U^T U=E$`，进而可以导出：
```mathjax!
$$
U \sum_{j=0}^K \alpha_j \Lambda^j U^T =\sum_{j=0}^K \alpha_j U\Lambda^j U^T = \sum_{j=0}^K \alpha_j L^j
$$
```
　　因此，(4)式可以写成：
```mathjax!
$$
y_{output}=\sigma \left( \sum_{j=0}^K \alpha_j L^j x \right) \qquad(5)
$$
```
　　其中 `!$({\alpha_1},{\alpha_2},\cdots,{\alpha_K}) $` 是任意的参数，通过初始化赋值然后利用误差反向传播进行调整（训练的过程）。
　　
　　式(5)所设计的卷积核其优点在于：
　　１、卷积核只有 K 个参数，一般 K 远小于 n。
　　２、矩阵变换后，不需要做特征分解，直接用拉普拉斯矩阵 L 进行变换，计算复杂度变成了 `!$\mathcal{O}(n)$`。
　　３、卷积核具有很好的 spatial localization，特别地，K 就是卷积核的receptive field，也就是说每次卷积会将中心顶点 K-hop neighbor上的 feature 进行加权求和，权系数就是 `!$\alpha_k$`。更直观地看， K=1 就是对每个顶点上一阶neighbor的feature进行加权求和，如下图所示：
  
![enter description here](./images/8.jpg)

　　同理，K=2的情形如下图所示：
  
![enter description here](./images/9.jpg)

**注：上图只是以一个顶点作为实例，GCN每一次卷积对所有的顶点都完成了图示的操作。**

#### 七、 利用Chebyshev多项式递归计算卷积核

在第二代GCN中，`!$L$` 是 `!$n\times n$` 的矩阵，所以 `!$L^j$` 的计算还是 `!$\mathcal{O}(n^2)$` 复杂的，[Wavelets on graphs via spectral graph theory](https://link.zhihu.com/?target=https%3A//www.sciencedirect.com/science/article/pii/S1063520310000552) 提出了利用Chebyshev多项式拟合卷积核的方法，来降低计算复杂度。卷积核 `!$g_{\theta}(\Lambda)$` 可以利用截断（truncated）的shifted Chebyshev多项式来逼近。（这里本质上应该寻找Minimax Polynomial Approximation，但是作者说直接利用Chebyshev Polynomial的效果也很好）
```mathjax!
$$
g_{\theta}(\Lambda) = \sum_{k = 0}^{K - 1} \beta_k T_k (\tilde \Lambda)
$$
```
`!$\beta_k$` 是Chebyshev多项式的系数。`!$T_k(\tilde \Lambda)$` 是取 `!$\tilde \Lambda = 2\Lambda / \lambda_{max} - I$` 的Chebyshev多项式，进行这个shift变换的原因是Chebyshev多项式的输入要在 `!$\left[ -1,1\right]$` 之间。

由Chebyshev多项式的性质，可以得到如下的递推公式：
```mathjax!
$$
T_k (\tilde{\Lambda})x = 2\tilde{\Lambda}T_{k-1} (\tilde{\Lambda})x-T_{k-2} (\tilde{\Lambda})x \qquad(6)   \\
T_{0} (\tilde{\Lambda}) = I,T_{1} (\tilde{\Lambda}) = \tilde{\Lambda}
$$
```
其中, `!$x$` 的定义同上，是 `!$n$` 维的由每个顶点的特征构成的向量（当然，也可以是 `!$n\times m$` 的特征矩阵，这时每个顶点都有 `!$m$` 个特征，但是 `!$m$` 通常远小于 `!$n$` ）。

**这个时候不难发现：式（6）的运算不再有矩阵乘积了，只需要计算矩阵与向量的乘积即可。计算一次 `!$T_k (\tilde{\Lambda})x$` 的复杂度是 `!$\mathcal{O}(\left| E \right|)$` ， `!$E$` 是图中边的集合，则整个运算的复杂度是 `!$\mathcal{O}(K\left  | E \right|)$` 。当graph是稀疏图的时候，计算加速尤为明显，这个时候复杂度远低于 `!$\mathcal{O}(n^2)$` 。**

上面的讲述是GCN最基础的思路，很多论文中的GCN结构是在上述思路的基础上进行了一些简单数学变换。理解了上述内容，就可以做到“万变不离其宗”。

#### 八、 在GCN中的Local Connectivity和Parameter Sharing

 **CNN中有两大核心思想：网络局部连接，卷积核参数共享。**
 
 这两点在GCN中是怎样的呢？以下图的graph结构为例来探究一下：
 
![enter description here](./images/10.png)

###### 1. **GCN中的Local Connectivity**




###### 2. **GCN中的Parameter Sharing**