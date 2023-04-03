# Volume Rendering

今天给大家讲的是计算机图形学中的体渲染技术。体渲染技术是 NeRF 中用到的一种基于物理的渲染。

## Gallery

首先我们来看看一些体渲染的例子。左边的体积云出自 2017 年 Siggraph 的 "Spectral and Decomposition Tracking for Rendering Heterogeneous Volumes"，是一篇从数学原理上加速体渲染的文章。右边的对比图出自 2019 年 Siggraph 的 "Volume Path Guiding Based on Zero-Variance Random Walk Theory"，是一篇基于零方差随机游走理论的体渲染文章。随后的一个视频是 2017 年那篇论文的 demo。需要指出的是，视频是离线（非实时）渲染。

## What Is Rendering?

看完了体渲染的一些例子，我们可以开始介绍体渲染了。首先得介绍什么是渲染。在 CG 里面，渲染是通过计算机程序从模型生成图像的过程。简单来说数据到图像的过程就叫渲染。渲染大致分为真实感渲染和风格化渲染。体渲染是一种基于物理的渲染（PBR）的方法。什么是 PBR？PBR 是用真实世界的物理模型来建模光的传播行为，然后生成更贴近真实世界的画面。由于人们很早就清楚了光的物理性质，因此，PBR 也是真实感渲染的最大的一个研究方向。PBR 中一大研究方向就是著名的光线追踪技术。

## Models of Light

光的模型一般有三种，按照对于光的物理性质描述的准确性，依次分为量子光学，波动光学，几何光学。量子光学的模型最准确，但是对于图像的生成来说尺度太小，细节太多，计算量巨大，因此只有特定的场合需要用到量子模型。然后是波动光学。光的本质是电磁波，因此在一般尺度下，这个模型能准确地描述光的行为，包括干涉，衍射，极化等现象。但是我们很少需要模拟这些现象，因此这个模型也很少用。一个以波动模型为基础描述光线追踪的文章是 A Generic Framework for Physical Light Transport。现行最流行的模型是几何光学模型，也是中学教的光的模型，能最简单地描述光的成像模型。光线对应于辐射能量的流动方向，在两点之间的传播路径满足最短时间传播，在我们所处的时空中，我们将光看作直线，因此光线方程就可以用直线方程简单地描述出来。如果是在宇宙中，大质量的黑洞附近，空间被扭曲，此时的路径就不是直线，而是空间中两点对应的测地线。这样的研究也早已存在，并且已经用到了电影画面中。

## Radiometry

除了几何光学里面对于光线和物体相互作用的描述，我们还需要定量计算相互作用发生时光强度的变化。辐射度量学是定量研究辐射传播的。其中有几个比较重要的概念。

1. 光通量：光通量是指单位时间内流过某个表面的光能，符号 $\Phi$，单位 $W$。反应在图上就是单位时间内从球上发出的光能。
2. 辐照度/辐射度：单位表面积的入射/出射辐射功率，符号 $E/B$，单位 $W/m^2$。反应在图上就是光通量除以球的表面积。
3. 辐射：从所有方向到达或离开表面某一点每单位立体角每单位投影面积的功率，符号 $L$，单位 $W/(m^2\cdot\text{rad})$。可以将辐射简单地看作空间中两点之间光子的传输量。反应在图上就是球上的某一点向某个指定的方向单位时间内发射或接收的能量。

什么是立体角？一个有限立体角 $\Omega$ 定义为半球上的一块区域的面积除以半球半径的平方。单位为 (sr)。请注意，立体角不取决于曲面 $A$ 的形状，而仅取决于总面积。为了计算空间中任意曲面或物体所围成的立体角，我们首先将曲面或物体投影到半球上，然后计算投影的立体角。

## Surface Rendering

有了上述的概念，我们就可以开始渲染了。这里我们简单地提一下表面渲染的相关概念。我们能看到一个物体的颜色，除了物体自己发的光，就是反射的光。因此我们需要一个物理量来描述反射。在最一般的情况下，光可以从入射方向 $\Psi$ 进入表面上的点 $\mathbf p$，并且可以从出射方向 $\Theta$ 离开表面上的点 $\mathbf q$ 。这里我们假设入射点和出射点是同一个点，因此我们可以得到双向反射分布函数（BRDF）。这个函数描述了当一束辐射打中某个物体的某一点时候，反射光的分布函数。对于某些点来说，反射光在各个方向上均匀分布，表现为漫反射；也可能集中在某一个方向，表现为镜面反射。

而在成像的时候，我们关心的是摄入相机所在方向的光线。因此为了计算从某一点发出的光强，我们既需要计算这一点直接发出的光线强度，也需要计算所有被反射到这个方向的光强。这就是表面渲染的方程的由来。特别的，由于光线可以反射不止一次，我们完全可以计算光线多次反射后的强度和颜色，即进行路径追踪。

## Participating Media

介绍完表面渲染，我们应该开始介绍体渲染了。之前的渲染公式里面，光线传播的模型被过度简化了。

1. 首先是最关键的假设，光线在传播时候，辐射度会保持不变。但其实这条假设只有光在真空中传播的时候才会成立。
2. 第二条假设是，光在击中物体时候，只会在击中点处发生反射、折射。但是对于透明的物体，光穿过物体并且在物体内部发生反射、折射。
3. 第三条假设是所有物体都有一个清晰的表面。但是对于云烟雾和火焰这种物体来说，清晰定义它们的表面比较困难。

针对第一点，我们在光传播的模型中引入传播介质。这些介质可以做到：

1. 将其他形式的能量转化为光能。这表现为介质的发光性质。
2. 将光能转化为其他形式的能量。这表现为介质的吸收性质。
3. 改变光线的传播方向。这表现为介质的散射性质。

针对第二三点，我们引入一种新表示方式，体表示。体表示就是把物体的物理性质以体像的形式存储，类似于图像。

体渲染是对光线穿过体像前后辐射变化的一种呈现。因此体渲染中，光线仍然可以看作是直线传播。具体的光线方程为：
$$
\mathbf{r}(t)=\mathbf{x}+t\Theta,\Theta=\frac{\mathbf{y}-\mathbf{x}}{\|\mathbf{y}-\mathbf{x}\|},0\le t\le s=\|\mathbf{y}-\mathbf{x}\|
$$

## Absorption

介质吸收最典型的例子是：光击中介质后会加热介质，即光能转化为内能。显然，一束光穿过吸收介质后的强度变化量，和介质，穿过介质的长度，初始光强有关。

描述介质在空间中某点的吸收强度的物理量叫做吸收系数 $\sigma_a(t)$​。通常来说，吸收相对于入射的方向是各向同性的。因此微分方程如下。
$$
\mathrm{d}L(z\to\Theta)=-\sigma_a(z)L(z\to\Theta)\mathrm{d}t
$$

## Emission

一般来说，介质（如火）发光的强度可以通过体积发射函数 $\epsilon(z)$（单位 [$W/m^3$]）来表征。它基本上告诉我们每单位体积和每单位时间在三维空间中的某个点 $z$ 发射了多少光子。通常，体积发射是各向同性的，这意味着在 $z$ 周围的任何方向发射的光子数等于 $\epsilon(z)/4\pi$（单位 [$W/m^3\cdot\text{sr }$]）。因此当光线穿过发光介质时候，在 $z$ 处无限小厚度 $\mathrm{d}t$ 的切片中的体积发射给 $\Theta$​ 方向增加的辐射为：
$$
\mathrm{d}L(z\to\Theta)=\frac{\epsilon(z)}{4\pi}\mathrm{d}t=L_e(z)\mathrm{d}t
$$

## Out-Scattering

外散射对于光线的效果和吸收对于光线的效果是完全一致的，即都是减少光线的强度。因此仿照吸收系数，我们也可以定义一个散射系数 $\sigma_s(t)$。吸收和散射系数的和被称作消光系数 $\sigma_t(t)$。

## In-scattering

$z$ 处的相位函数 $p(z,\Psi\leftrightarrow\Theta)$（单位 [$\mathrm{sr}^{-1}$]）描述了从方向 $\Psi$ 散射到 $\Theta$ 的概率。通常，相位函数仅取决于 $\Psi$ 和 $\Theta$ 两个方向之间的夹角。
$$
\mathrm{d}L(z\to\Theta)=\sigma_s(z)\int_{\mathcal{S}^2}p(z,\Theta\leftrightarrow\Psi)L(z\to\Psi)\mathrm{d}\omega_{\Psi}\mathrm{d}t\\
=\sigma_s(z)L_s(z\to\Theta)\mathrm{d}t
$$

## Radiative Transfer Equation

将上述介质的作用相加，我们就可以得到辐射传输方程。我挑选了近几年来 Siggraph 描述 Radiative Transfer Equation 最常见的形式。

值得注意的是，为了形式统一，而且发射强度和吸收系数都只和位置有关，我们给发射项乘上吸收系数。
$$
(\Theta\cdot\nabla)L(z\to\Theta)=-\sigma_t(z)L(z\to\Theta)+\sigma_a(z)L_e(z\to\Theta)+\sigma_s(z)L_s(z\to\Theta)
$$
首先我们解齐次微分方程：
$$
\begin{align*}
\frac{\mathrm{d}}{\mathrm{d}t}L(t)&=-\sigma_t(t)L(t)\\
\frac{\mathrm{d}L(t)}{L(t)}&=-\sigma_t(t)\mathrm{d}t\\
\ln L(t)-\ln L(0)&=-\int_0^t\sigma_t(u)\mathrm{d}u\\
L(t)&=L(0)\exp\left(-\int_0^t\sigma_t(u)\mathrm{d}u\right)\\
\end{align*}
$$

## Transmittance

这里我们能得到两个概念。

Transmittance 的概念是从表面渲染里面发展出来的，他的原本的定义是 $\Phi_t/\Phi_r$。这里是体渲染借用过来的概念，表示光线穿越一段距离后能剩余多少光强度。由于光线强度和光线中光子的数目成正比。因此从统计上来说， $T(0,t)$ 可以表示光子保留的百分比数；从概率上来说，$T(0,t)$ 可以表示光子在这段距离中不和介质发生作用的概率。

## Radiative Transfer Equation

$$
\begin{align*}
L(t)&=M(t)\exp\left(-\int_0^t\sigma_t(u)\mathrm{d}u\right)\\
\mathrm{d}M(t)&=\exp\left(\int_0^t\sigma_t(u)\mathrm{d}u\right)[\sigma_a(t)L_e(t)+\sigma_s(t)L_s(t)]\mathrm{d}t\\
M(t)-M(0)&=\int_0^t\exp\left(\int_0^u\sigma_t(v)\mathrm{d}v\right)[\sigma_a(u)L_e(u)+\sigma_s(u)L_s(u)]\mathrm{d}u\\
L(t)&=L(0)\exp\left(-\int_0^t\sigma_t(u)\mathrm{d}u\right)+\int_0^t\exp\left(-\int_u^t\sigma_t(v)\mathrm{d}v\right)[\sigma_a(u)L_e(u)+\sigma_s(u)L_s(u)]\mathrm{d}u\\
L(t)&=L(0)T(0,t)+\int_0^tT(u,t)[\sigma_a(u)L_e(u)+\sigma_s(u)L_s(u)]\mathrm{d}u
\end{align*}
$$

注意到 $L(0)=M(0)$。

## NeRF's Setting

$$
L(s)=L(0)T(0,s)+\int_0^sT(t,s)\sigma_a(t)L_e(t)\mathrm{d}t\\
T(0,t)=\exp\left(-\int_0^t\sigma_t(u)\mathrm{d}u\right)
$$

其中 $0$ 处是发光点，$s$​ 处是相机所在位置。
$$
\mathbf{r}(t)=\mathbf{x}+t\Theta,\Theta=\frac{\mathbf{y}-\mathbf{x}}{\|\mathbf{y}-\mathbf{x}\|},0\le t\le s=\|\mathbf{y}-\mathbf{x}\|
$$
NeRF 的设定里面，包括一般的光线追踪，光线是从相机出发的：

其中 $t_f$ 处是发光点，$0$ 处是相机所在位置。
$$
\mathbf{r}(t)=\mathbf{x}+t\Theta,\Theta=\frac{\mathbf{y}-\mathbf{x}}{\|\mathbf{y}-\mathbf{x}\|},0\le t\le s=\|\mathbf{y}-\mathbf{x}\|
$$

$$
C(\mathbf r)=\mathbf{c}(\infty,\mathbf{d})T(t_f)+\int^{t_f}_{t_n}T(t)\sigma(\mathbf r(t))\mathbf c(\mathbf r(t),\mathbf d)\mathrm dt\\
T(t)=\exp\left(-\int^{t}_{t_n}{\sigma(\mathbf r(s))}\mathrm ds\right)
$$

离散化积分公式：
$$
L(s)=L(0)T(0,s)+\int_0^sT(t,s)\sigma_t(t)L_e(t)\mathrm{d}t\\
T(t,s)=\exp\left(-\int_t^s\sigma_t(u)\mathrm{d}u\right)
$$
将 $0$ 到 $s$ 大致均等地划分为 $n$ 分，每段长度为 $\delta_i$，每段的采样为区间右端点 $x_i$，因此：
$$
\begin{align*}
T(0,s)&=\exp\left(-\int_0^s\sigma_t(u)\mathrm{d}u\right)\\
&=\exp\left(-\sum_{i=1}^n\sigma_t(x_i)\delta_i\right)\\
&=\prod_{i=1}^{n}\exp(-\sigma_t(x_i)\delta_i)
\end{align*}
$$

$$
\begin{align*}
\int_0^sT(t,s)\sigma_t(t)L_e(t)\mathrm{d}t&=\sum_{i=1}^n\int_{x_{i-1}}^{x_i}T(t,s)\sigma_t(t)L_e(t)\mathrm{d}t\\
&\approx\sum_{i=1}^n\int_{x_{i-1}}^{x_i}T(t,s)\sigma_t(t)L_e(x_i)\mathrm{d}t=\sum_{i=1}^nL_e(x_i)\int_{x_{i-1}}^{x_i}T(t,s)\sigma_t(t)\mathrm{d}t\\
&=\sum_{i=1}^nL_e(x_i)\int_{x_{i-1}}^{x_i}\frac{\mathrm{d}}{\mathrm{d}t}\exp\left(-\int_t^s\sigma_t(u)\mathrm{d}u\right)\mathrm{d}t=\sum_{i=1}^nL_e(x_i)\left[T(x_i,s)-T(x_{i-1},s)\right]\\
&=\sum_{i=1}^nL_e(x_i)T(x_i,s)\left[1-T(x_{i-1},x_i)\right]\approx\sum_{i=1}^nL_e(x_i)T(x_i,s)\left[1-\exp\left(-\int_{x_{i-1}}^{x_i}\sigma_t(x_i)\mathrm{d}u\right)\right]\\
&=\sum_{i=1}^nL_e(x_i)T(x_i,s)\left[1-\exp(-\sigma_t(x_i)\delta_i)\right]
\end{align*}
$$

因此我们可以定义：
$$
\alpha_i=1-\exp(-\sigma_t(x_i)\delta_i)
$$
对应的离散渲染公式就是：
$$
L(s)=L(0)\prod_{i=1}^{n}T(0,s)+\sum_{i=1}^nL_e(x_i)T(x_i,s)\alpha_i\\
T(x_i,s)=\prod_{j=i+1}^{n}\exp(-\sigma_t(x_j)\delta_j)=\prod_{j=i+1}^n(1-\alpha_i)
$$
