[TOC]

# Real-time Neural Radiance Caching for Path Tracing

~~今天我来传教了。~~今天带来的是 NGP 的前作之一 Neural Radiance Caching。这篇文章的目标是达到实时的路径追踪，用的方法是辐射缓存 Radiance Caching 和神经网络 Neural Network。在讲这篇文章的同时也会略微介绍 NRC 里面的 baseline，ReSTIR。这里我们假设大家稍微懂一点光追的知识，大概是 CG HW 3 的那种懂就行了。

## Path Tracing

NRC 的目标是加速 Path Tracing，这里简单的介绍一下什么是 Path Tracing。Path Tracing 是一种基于光线追踪（Ray Tracing）实现全局光照（Global Illumination）的算法。我们知道光线追踪算法完全基于渲染方程：
$$
L_o(p_1\to p_0)=L_e(p_1\to p_0)+\int_{\mathcal{H}^2}f(p_2\to p_1\to p_0)L_i(p_2\to p_1)|\langle\mathbf{n}_{p_1},\omega_i\rangle|\mathrm{d}\omega_i
$$
渲染方程描述了光线的传播。

仔细观察渲染方程，我们会将 $L_i$ 看作是从某个发光体发出的光。在现实世界中，$L_i$ 可以是由反射了其他光源发出的光的反射面发出的，因此我们可以将 $L_i$ 拆开成反射面直接发出的光和反射面反射的光。在 $p_2$ 这点，渲染方程的形式是完全一致的：
$$
L_i(p_2\to p_1)=L_e(p_2\to p_1)+\int_{\mathcal{H}^2}f(p_3\to p_2\to p_1)L_i(p_3\to p_2)|\langle\mathbf{n}_{p_2},\omega_{ii}\rangle|\mathrm{d}\omega_{ii}
$$
如此，我们就考虑了一次从 $p_2$ 到 $p_1$ 到相机的路径追踪。将上式带入 $\mathbf{x}$ 处的渲染方程：
$$
\begin{align*}
L_o(p_1\to p_0)&=L_e(p_1\to p_0)\\
&+\int_{\mathcal{H}^2}f(p_2\to p_1\to p_0)\biggl[L_e(p_2\to p_1)+\int_{\mathcal{H}^2}f(p_3\to p_2\to p_1)L_i(p_3\to p_2)|\langle\mathbf{n}_{p_2},\omega_{ii}\rangle|\mathrm{d}\omega_{ii}\biggr]|\langle\mathbf{n}_{p_1},\omega_i\rangle|\mathrm{d}\omega_i\\
&=L_e(p_1\to p_0)\\
&+\int_{\mathcal{H}^2}f(p_2\to p_1\to p_0)L_e(p_2\to p_1)|\langle\mathbf{n}_{p_1},\omega_i\rangle|\mathrm{d}\omega_i\\
&+\int_{\mathcal{H}^2}f(p_2\to p_1\to p_0)\int_{\mathcal{H}^2}f(p_3\to p_2\to p_1)L_i(p_3\to p_2)|\langle\mathbf{n}_{p_2},\omega_{ii}\rangle|\mathrm{d}\omega_{ii}|\langle\mathbf{n}_{p_1},\omega_i\rangle|\mathrm{d}\omega_i
\end{align*}
$$
改一下变量的名字。而且由于只有光源会发光，因此将积分从立体角上采样转为在光源上采样，这就是 Next Event Estimation 对于直接光照的处理。（看情况介绍什么是从立体角上采样转为在光源上采样）：
$$
\begin{align*}
L(p_1\to p_0)&=L_e(p_1\to p_0)\\
&+\int_{A}f(p_2\to p_1\to p_0)L_e(p_2\to p_1)G(p_2\leftrightarrow p_1)\mathrm{d}A(p_2)\\
&+\int_{A}f(p_2\to p_1\to p_0)\int_{A}f(p_3\to p_2\to p_1)L_e(p_3\to p_2)G(p_3\leftrightarrow p_2)G(p_2\leftrightarrow p_1)\mathrm{d}A(p_3)\mathrm{d}A(p_2)\\
\end{align*}
$$
这个过程可以无限进行下去，因此通项形式就是：
$$
L(p_1\to p_0)=\sum_{n=1}^{\infty}P(\bar{p}_n)\\
P(\bar{p}_n)=\underbrace{\idotsint}_{n-1}L_e(p_n\to p_{n-1})\left[\prod_{i=1}^{n-1}f(p_i\to p_{i-1}\to p_{i-2})G(p_i\leftrightarrow p_{i-1})\right]\mathrm{d}A(p_n)\cdots\mathrm{d}A(p_2)
$$

（这里得说一句，CG PPT 上的公式好像是错的……）

## Real-time Ray Tracing

Path Tracing 好是好，算全局光照没有比他更好的算法了，但是它的一个问题就是慢，不仅慢在光线与物体的求交，也慢在采样的数目要非常巨大。由于高维的积分没有什么很好的求解方法，我们只能使用 Monte Carlo 积分来得到积分的结果。对于没有任何技巧的暴力 Path Tracing 算法，采样数必须要非常大才可以收敛到一个合理的值。下图是每个像素采样数（Sample per pixel, spp）为 8 和 1024 的比较。

假设一张图 1920 * 1080，每个像素采样 1024 次，我们路径追踪 64 次，那一共要计算 1359 亿次颜色值（对比同分辨率的 NeRF 只需要计算 3.98 亿次颜色值）。就是因为这一点，Path Tracing 存在很多优化空间，加速积分收敛速度。

特别是今天要讨论的是实时光追的算法，实时光追的难点在于以下几点：

1. 所有的渲染工作必须在 33 毫秒内完成并且提交到屏幕上。
2. 一定要对 CUDA, GPU, 图形 API 比如 Vulkan 和并行计算非常非常熟悉。
3. 要渲染的场景是动态的，而渲染器一般不知道未来场景会如何变化，比较难加入辐射变化的先验信息。
4. 为了做到实时，每个像素只能追踪几条光线，而且路径深度很低，甚至只能做到直接光的采样。对于一个存在大量光源的场景，追踪到场景中所有灯的 shadow rays 是不可行的。所以我们需要一个非常高效的采样方法找到重要的灯光，而某一点上贡献最大的灯取决于每个灯对该点的可见度、散射函数（BSDF 或相位函数）在该点的分布、以及光源的功率和发光特性。

其实我们有很多方法应对这些问题：

1. 一个能尽可能减少求交查询次数的数据结构，比如 Displacement Mapping BVH。
1. 光追的精髓：找到并且完美地结合针对渲染方程各项（辐射度，BSDF，光源面积）最优的采样 PDF，这就是 ReSTIR 的做法。
1. 增加采样时间，用插帧算法补全中间帧；或者生成欠采样的图片，用超采样的方法得到一张比较高清的图片，这是现在很流行的 DLSS 的做法。
1. 减少路径追踪的数量的同时不造成有偏差的结果。这就是今天要讲的 NRC 的做法。

## ReSTIR & Radiance Caching

这里给大家简单介绍一下 NRC 里面出现的相关算法。

### ReSTIR

**直接光照**算法：ReSTIR。ReSTIR 全称是：Reservoir-based SpatioTemporal Importance Resampling，基于水池采样的时空重要性重采样，是一种不用神经网络的实时光追方法。这篇文章将适合 GPU 的采样方法和多采样 PDF 的 Monte-Carlo 积分方法结合起来，实现了大场景多光源的实时直接光追。

ReSTIR 的核心有 3 点：

1. 适合 GPU 的采样方法，水池采样。水池采样讲的是对于一串未知长度 $N$ 的数组，如何只遍历一遍就采样获得不重复的 $k$ 个数组元素？而且每个被采到的样本的概率是相同的，即 $k/N$。
2. 多采样 PDF 的 Monte-Carlo 积分方法，多重重采样 + 重要性重采样。一般的 Monte-Carlo 采样只考虑单一 PDF 的采样，如果有多个 PDF 可供采样，应该如何加权各种采样的结果？如果渲染函数很难被采样，但是光源比较容易采样，我们怎么把对光源的采样转移到对渲染函数的采样？
3. 空间上的，时间上的，可见度上的重复利用。假如是动态场景，那么我们可以合理地重复利用上一帧的采样，这必然能大大加快 Path Tracing 的收敛，那么怎么才是合理地重复利用呢？

ReSTIR 解决上述 3 个问题并且将答案完美地结合了起来。有兴趣的同学可以去看一下这篇文章。不过总体和 NRC 关联不大，只需要知道这是一个计算直接光照的算法即可。

### Irradiance Caching

1988 年 Ward 等人发现，漫反射场景中，虽然计算**间接光照**的辐照度（Irradiance，也就是所有方向朝这个点发射过来的光的能量）的成本很高，但在大多数场景中是平滑变化的。如右图所示。

因此我们可以针对场景计算它的 Irradiance，并且存储在一个开销较小的数据结构中，这步可以预计算，也可以渲染时计算。计算间接光照时候只需要直接根据周围缓存的插值即可得到一个相对正确的辐射值。详细可以搜索相关的资料，只需要知道这是一个加速间接光照的算法即可。总体效果可以说是差强人意。

---

间接光照计算公式：
$$
L_o(\mathbf{x},\omega_o)=\int_{\mathcal{H}^2}f(\mathbf{x},\omega_o,\omega_i)L_i(\mathbf{x},\omega_i)|\langle\mathbf{n}_{\mathbf{x}},\omega_i\rangle|\mathrm{d}\omega_i
$$
对于漫反射材质来说，BRDF 是 $\rho(\mathbf{x})/\pi$，因此 $\mathbf{x}$ 处的间接光照为：
$$
L_o(\mathbf{x},\omega_o)=\frac{\rho(\mathbf{x})}{\pi}\int_{\mathcal{H}^2}L_i(\mathbf{x},\omega_i)|\langle\mathbf{n}_{\mathbf{x}},\omega_i\rangle|\mathrm{d}\omega_i=\frac{\rho(\mathbf{x})}{\pi}E(\mathbf{x})
$$
后一个等号是 Radiance 的定义式。因此存储 Irradiance 并且合理地插值就可以得到场景中间接光照的变化。

---

### Radiance Caching

随后就是 Radiance Caching。Irradiance Caching 处理了漫反射的场景，那么对于有光泽（glossy）的场景应该怎么处理呢？一个直接的想法就是辐射场的概念，即记录空间中各个位置处各个方向的辐射信息，这样我们就有能力处理随着方向变化的辐射插值问题。具体做法这里就不介绍了。

其实即使不熟悉光追，大家也见过 Radiance Caching 的例子，比如 PlenOctree（值得一提的是，其实这正是 PlenOctree 渲染快的本质原因，但我觉得这也是正是它最后卷输了的原因。因为 Thomas 说，不必把 Grid 和 Neural Network 对立起来，要把二者结合起来并且发挥他们各自的优势）。

Radiance Caching 相关的技术还有 Photon Mapping。Photon Mapping 一般是处理 Bidirectional Path Tracing BDPT 问题，与 Radiance Caching 相比，它多了一步从光源到场景的前向光追过程，具体的算法就不介绍了。

有意思的是，去准备 Photon Mapping 算法的时候，我认为我找到了 Thomas 想到 Hash Encoding 的原因。PBRT 介绍的 Photon Mapping 算法需要把数据放到密集的 grid 中，由于 grid 的填充率并不高，因此可以用 hash table 代替 grid。

## Neural Radiance Caching

有了以上的铺垫，我们就可以正式开始介绍 NRC 了。

### Algorithm

NRC 的做法就是：渲染时候，用 Path Tracing 的算法计算前 $n-1$ 次反射所产生的辐射能量，在 $\mathbf{x}_n$ 处查询神经网络，即 NRC，来获得 $L_s(\mathbf{x}_n,\omega_n)$。通过使用神经网络来缓解定位、插值和更新缓存点的困难，因为众所周知，神经网络特别适合替换复杂的启发式算法。

就此展开，那么这里就产生了两个问题。

1. 究竟第几次反射时候停止 Path Tracing？
2. 神经网络应该怎么训练，是提前训练好了的，还是边渲染边训练？神经网络是如何对于不同场景进行泛化的？训练的策略是怎么样的？

### Path Termination

对于第一个问题，Thomas 这里直接借用了前人的概念：认为当辐射能量一旦面积散布变得足够大，就会模糊掉缓存的小规模不准确之处。对于所有路径，只要 $a(\mathbf{x}_1\cdots\mathbf{x}_n)>c\cdot a_0$，那么就停止 Path Tracing。如果这条路径被选为训练路径，那么在之前停止位置继续 Path Tracing，停止条件同上。为了能快速计算，总共选取的训练路径数量不多，只占总像素数的 2-3%.

### Online Adaption

对于第二个问题，这里的做法正是边渲染边训练。刚刚提到了训练路径的概念，就是神经网络的训练集。被选取为训练路径上的所有点上的特征都会被当作训练样本输入到网络当中。

至此，我们可以总结一下 NRC 的优缺点。

1. 相比于提前训练：完全支持相机、照明、几何和材质的任意动态，无需场景先验。
2. 相比于提前训练：特定案例的处理最终会导致复杂、脆弱的系统。缓存与材质和场景几何无关。独立于场景复杂性的特性使得运行时的开销和内存占用仅与选取的训练路径数有关，因此其性能不会随着场景复杂度的变化而出现很大波动。
3. 相比于 Path Tracing：NRC 用 bias 换取了低 variance。
4. 由于缓存与材质和场景几何无关，这套算法也可以用于 volume rendering 上，甚至大开脑洞一下，我觉得点云场景都可以用。

但是缺点也有：

1. 训练路径的最后一个顶点可能会到达辐射度缓存没有训练过的场景位置，这可能会产生较大的近似误差。这个只能通过尽可能的增大训练集来解决。
2. 迭代优化可能只模拟了多反弹照明的一个子集，而不是所有的光传输。这个问题 Thomas 让部分的训练路径用 Russian roulette 的方法终止，使得这部分的结果是无偏的。

总体来说，这篇文章的核心正是用与场景解耦的神经网络替换掉繁琐的 Cache 结构，实现了实时光追。

### Details

接下来就是一些细节。

#### Input Encoding

我们知道散射辐射函数 $L(x,\omega)$ 是个高度非线性，变化非常剧烈的函数，把 $x,\omega$ 当作输入直接用神经网络去拟合是拟合不好的。我个人觉得这里就产生了两条路径去解决这个问题。一个是类似 NeRF 的方法，不是直接拟合场景的辐射，而是拟合了能影响辐射的变量 $\sigma,c$，用这两个变化不是那么剧烈的变量去最终生成场景的辐射。还有一个就是 NRC 的做法，既然把 $x,\omega$ 当作输入不够好，那我就把场景中所有信息都输入进去，于是他把法向量 $n$，和 BSDF 有关的信息（表面粗糙度 $r$，漫反射系数 $\alpha$，镜面反射系数 $\beta$）都输入进了网络。个人觉得这种加先验的方法有点暴力，但是适合这种信息量比较大的 Path Tracing case。而且的确信息越多，拟合效果会越好。

虽然说我们有两条路径去解决拟合问题，但是对于 Coordinate-based 神经网络来说，有一个步骤那是必不可少的。就是对输入进行 Encoding。这里作者对于各个分量都进行了一些变换，把总共 14 维的输入变成了 64 维的输入。但是具体的 Encoding 方法就不介绍了，因为反正最后都换成了 Hash Encoding 那套。不过在这里 Thomas 还是做了很多的试验，有兴趣可以去看 NRC 和 NGP 对应的章节。

#### Rendering

如果以上问题都解决了，那么其实已经完成了。但是这时候发现，渲染的时候场景会闪动，原因是为了快速收敛，训练时候学习率很大，加上训练样本的噪声其实非常非常大，很干扰网络的收敛。为此作者提出，把网络权重在时间序列上过一个低通滤波，即指数移动平均。当前网络的权重和上一时刻的网络的权重做线性的加和，得到的新的权重组成的网络去作为真正用于渲染的网络，这样有效地缓解了闪动问题。

### Result

这里展示了 1spp 下 NRC 的速度和渲染结果，特别注意的是，增加了一步算法之后，帧率反而变高了。

这里展示了同渲染质量下 NRC 提升的效率和作为离线渲染器时候的质量。

这里展示了把 NRC 作为 direct illumination 的效果，可以看到 NRC 近似拟合了场景的 global illumination。特别观察这些条纹状的 artifacts。这里作者指出这就是 positional encoding 特有的 artifacts，当时他也试了其他的 encoding 但是效果都不好。

这里展示了对比另一篇 Neural Cache 的文章的效果。可以看到 DDGI 方差更小但是偏差更大，速度更快；而 NRC 方差较大但是偏差更小，速度稍慢；Path tracing 方差巨大但是无偏，速度巨慢。

### Fully Fused Neural Networks

说实话一开始看 NRC 的目的当然是看 TCNN 的架构是怎么设计的。Pytorch/Tensorflow 比 CUDA 慢的原因是在于内存访问。对于 Pytorch/Tensorflow 来说，为了比较好的通用性，GPU 上的数据只能放在显存上。而 CUDA 可以有更多定制化的设计，可以把数据放在 Cache，Shared Memory 甚至 Register 里面，待到合适的时候再放到显存上。这就是 TCNN 的设计目标，加快内存访问。整个 TCNN 是针对于 RTX 3090 这个显卡进行设计的，因此对于其他型号的显卡，如果要充分展现 GPU 的计算能力，TCNN 的相关设定可能需要稍微修改一下。

Fully fused 的含义就是把整个神经网络作为一个 GPU 的核函数，即除了网络的输入输出、网络权重的读取是需要和显存交互，其他都是在与 Cache，Shared Memory 等更快速的内存层级交互。

最左边的图表示了神经网络运算的方式，其中 $N=16384,M_{\text{in}}=64,M_{\text{hidden}=64},M_{\text{out}=3}$。因此输入矩阵是 $64\times16384$ 的矩阵，除了最后一层外每层的权重都是一个 $64\times64$ 的矩阵，大小为 $16\mathrm{KB}$，能被寄存器完全容纳住。中间的图表示在计算的时候，输入矩阵分解为 $64\times128$ 的小块，CUDA 的每个 block 去读取对应的小块的数据，然后进行矩阵计算并且过激活函数，得到 $64\times128$ 的中间结果。这个中间结果只有 $32\mathrm{KB}$，完全能塞入 shared memory 里面。为了进一步加速矩阵计算，这里再把矩阵分解为 $16\times16$ 的小块，使得矩阵运算能完全用半精度的 TensorCore 硬件加速。这就是 TCNN 做到的矩阵运算加速的技巧。

这里很喜欢 Thomas 的一句话，原句出自今年他自己在 nVidia 介绍 NGP 的 presentation 里面：那么这是否意味着我们都应该停止使用 Python 而改用低级别的语言呢？当然不是。快速构建模型有它明显的优势。但是！我在这个项目中的经验是，低级别的开发开销比我最初想象的要少得多，而且在某些方面实际上允许更多的灵活性，因为你可以深入几乎所有东西的内部。而且，很明显，即使你花了更多的时间来编程，当训练时间以秒为单位而不是以小时为单位时，这也可以得到补偿。因此，虽然我们还不应该放弃我们的框架，但我认为值得对低级别的方法给予更多的考虑，而不是你通常会做的。（So does this mean we should all stop using Python and go low-level instead? Of course not. Being able to prototype quickly has obvious advantages. But! My experience with this project has been that going low-level is much less of a development overhead than I initially thought and in some ways actually allows for more flexibility, because you can go under the hood of pretty much everything. And, obviously, even if you spend more time programming, this can be compensated when training times are measured in seconds instead of hours. So while we shouldn’t drop our frameworks just yet, I think it's worth giving the low-level approach more consideration than you would typically do.）

## Conclusion

优点：

1. 快，是真的快，特别是 NGP 继承了这里减少迭代次数的思想，同时点名表扬 TCNN。
2. 与场景解耦，留给了后人很多优化的空间。

缺点：

1. 还是继承了 Cache 会漏光的缺点，特别是神经网络没收敛的时候，但是看不大出来。
2. 焦散做不出来，因为本质还是反向光追。不过问题不大，加个前向光追的过程就可以了。

