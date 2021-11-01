# Causal Inference and Uplift Modeling A review of the literature论文笔记

# 摘要

  Uplift模型是对客户施加一个行动或者策略引起的增量效果而进行建模的一套技术。Uplift模型既是一个因果问题，也是一个机器学习问题。关于uplift的文献主要分为3大类：Two-Model、类别转换的方式、直接对uplfit建模。不幸的是，在缺乏因果推断和符号表示的通用框架的情况下，这三种方法很难评估。在这片文章，我们使用Rubin关于因果推断的模型和它对计量经济学的表示法对这三种方法进行了清晰的比较，并对其中一种进行概况。据我们所知，本文是第一篇关于uplift模型的综述文献。此外，本文表明，在极限的情况下，最小化因果效果估计量的MSE等同于最小化修改后的目标变量来替换未被观察到的Treatment效果的MSE。最后，我们希望本文在商业领域(或者医学、社会学、经济学)中将机器学习应用在因果推断的有兴趣的研究人员有用。

# 1 介绍

  Uplift模型是指公司可以评估客户的增益效果的一系列模型，即，评估一个行为对某些客户的影响。例如，一个电信公司的经理可能有兴趣评估向不同的客服发送促销邮件的效果，来了解他们在下一个时期对更换电话套餐的倾向。有了这些信息，这个经理就可以有效的锁定客户。
  评估客户的增益效果既是一个因果推断问题也是一个机器学习问题。这是一个因果问题i因为这需要估计一个人收到互斥行为的两个action的结果差异（一个人要么收到促销邮件要么没有收到，不能同时收到或者不收到）。为了克服这种反事实的性质，uplift模型强烈依赖于随机试验，即将客户随机分为被施加策略（treatment组，实验组）或者没有被施加策略（对照组）。uplift模型同样是一个机器学习问题，因为它需要训练不同的模型并且根据一些性能指标选择最可靠的模型。这也需要交叉验证策略以及潜在的特征工程。
  Uplift 模型文献提到了3种主要的方法将因果推断方面与机器学习进行结合。这导致了不同的uplif指标和评估方式，这不容易互相比较。我们将这个问题归为一下原因：研究人员未使用通用的框架和符号表示。
  在这文章中我们采用Rubin提出的模型为共同参考框架对这三种方法（Two-model、类别转换方法、直接建模方法）进行描述。我们尝试采用系统化的“机器学习”来建立预测模型。本文的目标是提供了一个通用的框架，来统一不同的uplfit方法，从而使得他们的比较和评估变得更加容易。最后，我们的文献表明，在极限的情况下，最小化uplift效果的MSE等同于最小化用一个修正的目标变量替代未被观测到的uplift效果的MSE。

# 2 因果推断：基础

  我们依赖Rubin提出的关于因果推断的模型并使用计量经济学文献的标准符号。这个模型的核心是对潜在的结果和因果效应进行符号化。我们用i表示N个个体的索引。![Y_i(1)](https://math.jianshu.com/math?formula=Y_i(1))表示的第![i](https://math.jianshu.com/math?formula=i)个人受到treatment的结果，![Y_i(0)](https://math.jianshu.com/math?formula=Y_i(0))表示的第![i](https://math.jianshu.com/math?formula=i)个人没有收到任何treatment的结果（就是对照组的结果）。**因果效果用 ![\color{red}{\tau_i}](https://math.jianshu.com/math?formula=%5Ccolor%7Bred%7D%7B%5Ctau_i%7D)表示**,表示是treatment和control组的差异，即
 ![\tau_i = Y_i(1) -Y_i(0) \tag{1}](https://math.jianshu.com/math?formula=%5Ctau_i%20%3D%20Y_i(1)%20-Y_i(0)%20%5Ctag%7B1%7D)
 研究人员通常估计的是**CATE**（ Conditional Average Treatment Effect）,也就是说，对人群中的一个子人群来评估因果关系（用一个自人群的因果效果来表示一个单个人的因果效果）。
 ![CATE: \tau_i= E[Y_i(1)|X_i] -E[Y_i(0)|X_i] \tag{2}](https://math.jianshu.com/math?formula=CATE%3A%20%5Ctau_i%3D%20E%5BY_i(1)%7CX_i%5D%20-E%5BY_i(0)%7CX_i%5D%20%5Ctag%7B2%7D)
 这个![X_i](https://math.jianshu.com/math?formula=X_i)是![L \times 1](https://math.jianshu.com/math?formula=L%20%5Ctimes%201)的特征向量。当然我们不可能同时观察到![Y_i(1)](https://math.jianshu.com/math?formula=Y_i(1))和![Y_i(0)](https://math.jianshu.com/math?formula=Y_i(0))。![W_i \in 0,1](https://math.jianshu.com/math?formula=W_i%20%5Cin%200%2C1)表示是否收到treatment，![W_i=1](https://math.jianshu.com/math?formula=W_i%3D1)表示一个人收到treatment，![W_i=0](https://math.jianshu.com/math?formula=W_i%3D0)表示一个没有收到treatment，那么一个人![i](https://math.jianshu.com/math?formula=i)的结果可以用以下公式表示:
 ![Y_i^{obs}= W_iY_i(1) +(1-W_i)Y_i(0) \tag{3}](https://math.jianshu.com/math?formula=Y_i%5E%7Bobs%7D%3D%20W_iY_i(1)%20%2B(1-W_i)Y_i(0)%20%5Ctag%7B3%7D)
 一个流行然而不幸错误的观点是，人们可以通过简单的计算公式4来表估测CATE
 ![E[Y_i^{obs}|X_i=x,W_i=1] - E[Y_i^{obs}|X_i=x,W_i=0] \tag{4}](https://math.jianshu.com/math?formula=E%5BY_i%5E%7Bobs%7D%7CX_i%3Dx%2CW_i%3D1%5D%20-%20E%5BY_i%5E%7Bobs%7D%7CX_i%3Dx%2CW_i%3D0%5D%20%5Ctag%7B4%7D)
 这个公式不能表示CATE，除非一个用户在![X_i](https://math.jianshu.com/math?formula=X_i)的条件下![W_i](https://math.jianshu.com/math?formula=W_i)要独立于![Y_i(1)](https://math.jianshu.com/math?formula=Y_i(1))和![Y_i(0)](https://math.jianshu.com/math?formula=Y_i(0))。这个假设就是社会学和医学中的**![\color{red}{CIA}](https://math.jianshu.com/math?formula=%5Ccolor%7Bred%7D%7BCIA%7D),Conditional Independence Assumption**假设。
 这个假设就是说在一个确定的特征![X_i](https://math.jianshu.com/math?formula=X_i)下用户是随机分到treatment组和对照组。
 ![CIA:\{Y_i(1),Y_i(1)\} \perp W_i|X_i](https://math.jianshu.com/math?formula=CIA%3A%5C%7BY_i(1)%2CY_i(1)%5C%7D%20%5Cperp%20W_i%7CX_i)
 定义倾向得分符号![p(X_i)=P(W_i=1|X_i)](https://math.jianshu.com/math?formula=p(X_i)%3DP(W_i%3D1%7CX_i)),这个表示的是在![X_i](https://math.jianshu.com/math?formula=X_i)条件下是进入treatment组的概率。

# 3 Uplift模型

  一般公司关注Uplift模型是为了对一些顾客施加一个action后的影响。例如，一个健身房老板有兴趣评估对特征为![X_i](https://math.jianshu.com/math?formula=X_i)的顾客发送电子邮件，来增加他们在下一个时期续约的可能性。换句话说，uplift模型等于评估CATE。尽管公司可以轻松的设置随机试验来确保CIA假设，事实上我们不可能观察到真正的![\tau_i](https://math.jianshu.com/math?formula=%5Ctau_i),使得可以用标准的监督学习来预估![\tau_i](https://math.jianshu.com/math?formula=%5Ctau_i)。*换句话说就是无法得到真正的label，造成监督学习无法进行*。如果我们假设![\tau_i](https://math.jianshu.com/math?formula=%5Ctau_i)是可以观察到的，那么我们可以简单的将数据集划分为训练集和测试集，并使用许多可用的算法来预估的![CATE \hat{\tau}(X_i)](https://math.jianshu.com/math?formula=CATE%20%5Chat%7B%5Ctau%7D(X_i))值，最小化训练集上损失函数，之后也就可以在测试集上用AUC 、F1 score、准确率来评估效果。
  在缺少真正的![\tau](https://math.jianshu.com/math?formula=%5Ctau)情况下，uplfit文献有三种方案来评估![{\tau}(X_i)](https://math.jianshu.com/math?formula=%7B%5Ctau%7D(X_i))。第一种是Two-model形式，它包含了两个预测模型，一个是在treatment组上训练模型，一个是在control组上训练模型。第二种是类别转换的方式。第三种就是修改已有的学习器(决策树、随机森林、svm等)来进行直接对uplift建模。

## 3.1  Two-model方法

  Two-model方法已经在很多论文中应用了，经常做完基准模型。这种方式是分别在treatment组和control组数据上对![E[Y_i(1)|X_i]](https://math.jianshu.com/math?formula=E%5BY_i(1)%7CX_i%5D) 和![E[Y_i(0)|X_i]](https://math.jianshu.com/math?formula=E%5BY_i(0)%7CX_i%5D)建模。Two-model的优点就是方法简单，因为推断是在treatment组和control组中分别进行的，这样无论是回归还是多分类任务都可以直接使用随机森林、XGBoost来进行训练。在两个组(实验组和对照组)单独的可以达到好的预测结果。然而，出于uplift的目的，尽管人们这种方法看似不错，但是一些作者表明这种方法通常不如其他的方法好。一个原因是这个2个模型以各自的结果进行预测，因此弱化了“uplift”的信号。*换句话说，就是Two-model方法是以各自数据集上的输出进行建模，而我们要学习的是两个数据集上的增益效果*。

## 3.2 类别转换的方法

  类别转换的方式是针对二分类的情境下提出的。这种方法的目标函数如下：
 ![Z_i=Y_i^{obs}W_i+(1-Y_i^{obs})(1-W_i) \tag{6}](https://math.jianshu.com/math?formula=Z_i%3DY_i%5E%7Bobs%7DW_i%2B(1-Y_i%5E%7Bobs%7D)(1-W_i)%20%5Ctag%7B6%7D)
 新的目标![Z_i](https://math.jianshu.com/math?formula=Z_i)为1时包含了2种情况 。1)在treatment组中并且![Y_i^{obs}=1](https://math.jianshu.com/math?formula=Y_i%5E%7Bobs%7D%3D1)， 2）在control组中并且![Y_i^{obs}=0](https://math.jianshu.com/math?formula=Y_i%5E%7Bobs%7D%3D0) 。除了这2种情况![Z_i](https://math.jianshu.com/math?formula=Z_i)应该为0。

> ![Z = \left \{ \begin{aligned} 1 &&if\ {W=1\ and \ Y=1} \\ 1 &&if \ {W=0 \ and \ Y=0} \\ 0 && other\\ \end{aligned} \right.](https://math.jianshu.com/math?formula=Z%20%3D%20%5Cleft%20%5C%7B%20%5Cbegin%7Baligned%7D%201%20%26%26if%5C%20%7BW%3D1%5C%20and%20%5C%20Y%3D1%7D%20%5C%5C%201%20%26%26if%20%5C%20%7BW%3D0%20%5C%20and%20%5C%20Y%3D0%7D%20%5C%5C%200%20%26%26%20other%5C%5C%20%5Cend%7Baligned%7D%20%5Cright.)

在个体被分到实验组和对照组的概率应该一样的假设下，Jaskowski and Jaroszewicz证明了以下结论：
 ![\tau(X_i) = 2P(Z_i=1|X_i)-1 \tag{7}](https://math.jianshu.com/math?formula=%5Ctau(X_i)%20%3D%202P(Z_i%3D1%7CX_i)-1%20%5Ctag%7B7%7D)

> ![\tau(X_i) = P(Z_i=1|X_i) - P(Z_i=0|X_i) \\ =P(Z_i=1|X_i) - (1-P(Z_i=1|X_i)) \\ = 2P(Z_i=1|X_i)-1](https://math.jianshu.com/math?formula=%5Ctau(X_i)%20%3D%20P(Z_i%3D1%7CX_i)%20-%20P(Z_i%3D0%7CX_i)%20%5C%5C%20%3DP(Z_i%3D1%7CX_i)%20-%20(1-P(Z_i%3D1%7CX_i))%20%5C%5C%20%3D%202P(Z_i%3D1%7CX_i)-1)

Uplift模型就等于训练![P(Z_i=1|X_i)](https://math.jianshu.com/math?formula=P(Z_i%3D1%7CX_i))。类别转换的方法之所以流行是因为它同样也是简单，但是效果要比Two-model更好一些，至今为止的分类模型都可以用到类别转换的方法。然而这有2个假设（二分类情景；个体对实验组和对照组的倾向性必须一样）看似太严格了。幸运的是，可以通过一些转换操作来解决实验组和对照组的个人倾向不同的问题![Y^*](https://math.jianshu.com/math?formula=Y%5E*)是对CATE的预估值，
 ![Y_i^*= Y_i(1) \frac{W_i}{\hat{p}(X_i)} - Y_i(0) \frac{(1-W_i)}{(1-\hat{p}(X_i)} \tag{8}](https://math.jianshu.com/math?formula=Y_i%5E*%3D%20Y_i(1)%20%5Cfrac%7BW_i%7D%7B%5Chat%7Bp%7D(X_i)%7D%20-%20Y_i(0)%20%5Cfrac%7B(1-W_i)%7D%7B(1-%5Chat%7Bp%7D(X_i)%7D%20%5Ctag%7B8%7D)
 这里![\hat{p}(x)](https://math.jianshu.com/math?formula=%5Chat%7Bp%7D(x))表示的是倾向得分。这种转换必须是在CIA假设下，在![X_i](https://math.jianshu.com/math?formula=X_i)的条件假设下等同于CATE。
 ![E[Y_i^* | X_i] = \tau(X_i) \tag{9}](https://math.jianshu.com/math?formula=E%5BY_i%5E*%20%7C%20X_i%5D%20%3D%20%5Ctau(X_i)%20%5Ctag%7B9%7D)
 注意到，是在完全随机试验和![Y^{obs}](https://math.jianshu.com/math?formula=Y%5E%7Bobs%7D)二分类结果条件下，我们可以根据公式3和公式8，进行对公式6改写：
 ![Z_i=\frac{1}{2}Y_i^*+(1-W_i) \tag{10}](https://math.jianshu.com/math?formula=Z_i%3D%5Cfrac%7B1%7D%7B2%7DY_i%5E*%2B(1-W_i)%20%5Ctag%7B10%7D)

> 怎么由公式3 、8改写的公式6？

最后，也可以把样本的权证考虑进去：
 ![P(Z_i=1) \frac{1}{N}[\sum_i(1_{Z_i=1})] - P(Z_i=0) \frac{1}{N}[\sum_i(1_{Z_i=0})] \tag{11}](https://math.jianshu.com/math?formula=P(Z_i%3D1)%20%5Cfrac%7B1%7D%7BN%7D%5B%5Csum_i(1_%7BZ_i%3D1%7D)%5D%20-%20P(Z_i%3D0)%20%5Cfrac%7B1%7D%7BN%7D%5B%5Csum_i(1_%7BZ_i%3D0%7D)%5D%20%5Ctag%7B11%7D)

## 3.3 直接Uplift建模

  这种方式是修改现在已有的学习器结构进行直接对uplift建模。有修改LR、k-nearest neighbors、SVM，比较流行的还是修改树学习器。在这个章节中主要介绍的是基于树的方法，并讨论树的分裂标准。
  这里我们定义是随机时间，即对任意特征![x](https://math.jianshu.com/math?formula=x)倾向得分![P(X_i=x)= \frac{1}{2}](https://math.jianshu.com/math?formula=P(X_i%3Dx)%3D%20%5Cfrac%7B1%7D%7B2%7D)，评估ATE的效果![\hat{\tau}](https://math.jianshu.com/math?formula=%5Chat%7B%5Ctau%7D)为：
 ![\hat{\tau}= \underbrace{ \frac{\sum_i{Y_i^{obs}W_i}}{ \sum_iW_i} }_{p} - \ \underbrace{ \frac{\sum_i{Y_i^{obs}(1-W_i)}}{ \sum_i(1-W_i)} }_{q} \tag{12}](https://math.jianshu.com/math?formula=%5Chat%7B%5Ctau%7D%3D%20%5Cunderbrace%7B%20%5Cfrac%7B%5Csum_i%7BY_i%5E%7Bobs%7DW_i%7D%7D%7B%20%5Csum_iW_i%7D%20%7D_%7Bp%7D%20-%20%5C%20%5Cunderbrace%7B%20%5Cfrac%7B%5Csum_i%7BY_i%5E%7Bobs%7D(1-W_i)%7D%7D%7B%20%5Csum_i(1-W_i)%7D%20%7D_%7Bq%7D%20%5Ctag%7B12%7D)

> 简单的讲，公式12就是用treatment的平均结果减去control的平均结果，p和q分别表示也是如此

  第一种分裂标准是求两个叶子节点的区别：
 ![\Delta = |\hat{\tau}_{Left} -\hat{\tau}_{Right} | \tag{13}](https://math.jianshu.com/math?formula=%5CDelta%20%3D%20%7C%5Chat%7B%5Ctau%7D_%7BLeft%7D%20-%5Chat%7B%5Ctau%7D_%7BRight%7D%20%7C%20%5Ctag%7B13%7D)
 分别在左右叶子节点上用公式12求出![\hat{\tau}_{Left}](https://math.jianshu.com/math?formula=%5Chat%7B%5Ctau%7D_%7BLeft%7D)和![\hat{\tau}_{Right}](https://math.jianshu.com/math?formula=%5Chat%7B%5Ctau%7D_%7BRight%7D)

  Rzepakowski and Jaroszewicz基于信息论提出另外一种分裂标准：
 ![\Delta_{gain}= D_{after\_split}(P^T,P^C) - D_{before\_split}(P^T,P^C) \tag{14}](https://math.jianshu.com/math?formula=%5CDelta_%7Bgain%7D%3D%20D_%7Bafter%5C_split%7D(P%5ET%2CP%5EC)%20-%20D_%7Bbefore%5C_split%7D(P%5ET%2CP%5EC)%20%5Ctag%7B14%7D)
 这里![D(.)](https://math.jianshu.com/math?formula=D(.))是差异度量，![P^T](https://math.jianshu.com/math?formula=P%5ET)、![P^C](https://math.jianshu.com/math?formula=P%5EC)分别是treatment组合control组的概率分布。因此，这个分裂的表示是分裂后的发散增益程度。对于![D(.)](https://math.jianshu.com/math?formula=D(.))有以下三种方式Kullback, Euclidean、Chi-Squared：
 ![KL(P:Q) = \sum_{k=left,right} p_k \log \frac{p_k}{q_k}](https://math.jianshu.com/math?formula=KL(P%3AQ)%20%3D%20%5Csum_%7Bk%3Dleft%2Cright%7D%20p_k%20%5Clog%20%5Cfrac%7Bp_k%7D%7Bq_k%7D)
 ![E(P:Q) = \sum_{k=left,right} (p_k-q_k)^2](https://math.jianshu.com/math?formula=E(P%3AQ)%20%3D%20%5Csum_%7Bk%3Dleft%2Cright%7D%20(p_k-q_k)%5E2)
 ![{\chi}^2 (P:Q) = \sum_{k=left,right} \frac{(p_k-q_k)^2}{q_k}](https://math.jianshu.com/math?formula=%7B%5Cchi%7D%5E2%20(P%3AQ)%20%3D%20%5Csum_%7Bk%3Dleft%2Cright%7D%20%5Cfrac%7B(p_k-q_k)%5E2%7D%7Bq_k%7D)

  在因果树模型中，Athey and Imbens提出以下分裂标准：
 ![\Delta = \frac{1}{\#children} \sum_{k=1}^{\#children} {\hat{\tau}_k}^2 \tag{15}](https://math.jianshu.com/math?formula=%5CDelta%20%3D%20%5Cfrac%7B1%7D%7B%5C%23children%7D%20%5Csum_%7Bk%3D1%7D%5E%7B%5C%23children%7D%20%7B%5Chat%7B%5Ctau%7D_k%7D%5E2%20%5Ctag%7B15%7D)  
 可以发现，当二分类结果时，公式15等价于上述的欧式分裂标准方式 。

   值得注意的事，Athey and Imbens提出一种“诚实”的方法，没有使用相同的数据进行生成树并估计叶子之间的uplift。而是把数据分为两部分，一部分用于生成树结果，另外一部分用于估算叶子节点的uplift值。

# 4 评估

   在这个章节中介绍如何评估Uplift模型的效果 和使用“计量经济学”符号进行推导。uplift模型的评估与传统的机器学习评估方式不同。在机器学习中，标准的评估方式是采用交叉验证的方式：把数据集划分为训练集和测试集，训练集上进行train模型，在测试集上进预测并与真实情况进行比较。在uplift模型中，交叉验证仍然是一个有效的方法，但是我们获得真实的uplift增益，因为我们无法在一个人身上同时观察到treat和不被treat的效果。
   为了说明这个指标，我们使用4.4章节中的模拟数据。这个数据包括1w个样本，其中4997是treatment数据，5003是control数据。目标值是二值结果。这里有个强烈的负面影响，因为25%的数据被描述为“Sleeping Dog”行为。这里使用19个分类特征 和80/20交叉验证。

> “Sleeping Dog”行为是指在treatment中这个个体起到负面影响，但是在control中不为如此。这里treatment中25%的人起到负面影响。*这里个人觉得 ：就是指因为你的treatment行动，惹怒了一些人，造成原本想买的人又拒绝去买了*

   这里实现了上面的三种方法。对于Two-model方法使用的事两个boosted树,而类别转换方法使用的是随机森林的方法。为了测试第三种方法，我们使用[uplift软件包](https://links.jianshu.com/go?to=%5Bhttps%3A%2F%2Fcran.r-project.org%2Fweb%2Fpackages%2Fuplift%2Findex.html%5D(https%3A%2F%2Fcran.r-project.org%2Fweb%2Fpackages%2Fuplift%2Findex.html))里的随机森林和因果条件推理以及[R语言包](https://links.jianshu.com/go?to=https%3A%2F%2Fgithub.com%2Fsusanathey%2FcausalTree.git)中的因果树。本文没对各种模型进行调超参，因为本文的目的是为了不同的方法，而不是要说明其中一种方法比其他的方法好。

## 4.1 传统的uplift指标

   在uplift场景中有个问题就是无法同时观察到一个个体的施加action和不施加action的结果，这样无法获取到真正的uplfit增益。所以很多uplfit文献中都是uplift bins 或者uplift curves来衡量效果。评估uplift效果的一种常见的做法就是首先对treatment组合control的数据分别进行预测，然后分别计算各种组内的每个分位数的平均值。再然后对相同的分位值取他们(实验组和对照组)之间差异。因此就可以得到每一个十分位之间的差异了。例如，如fig4表示，Two-Model中第一个十分位uplfit值为0.3，类别转换的防脏中第一个十分位的uplfit值也为0.3。这种方式尽管有用，但是无法进行不同模型之间的对比。（因为是无法衡量a、b两个图之间那个更好。）

![img](https://upload-images.jianshu.io/upload_images/8374736-ec2f6fcb71982816.png?imageMogr2/auto-orient/strip|imageView2/2/w/1200)

 为了有个更清晰的概念，我们话了fig2(a)这样图，最左边的条形图是对应的前10%的uplfit平均值，第二个条形图是前20%的uplift平均值。表现好的模型会在第一个分位值中的值更大一些，而在较大分位值中有个整体较小的值。*(能表现出依次递减的趋势是好的模型)* 。

![img](https:////upload-images.jianshu.io/upload_images/8374736-a623ebb90e56e904.png?imageMogr2/auto-orient/strip|imageView2/2/w/1200)

累计效果图

。
 最后我们看一下累计增益图:计算gain时要把uplfit乘以每个组的人数。



![(\frac{Y^T}{N^T} - \frac{Y^C}{N^C} )(N^T+N^C) \tag{16}](https://math.jianshu.com/math?formula=(%5Cfrac%7BY%5ET%7D%7BN%5ET%7D%20-%20%5Cfrac%7BY%5EC%7D%7BN%5EC%7D%20)(N%5ET%2BN%5EC)%20%5Ctag%7B16%7D)

这里![Y^T](https://math.jianshu.com/math?formula=Y%5ET) 、![N^T](https://math.jianshu.com/math?formula=N%5ET)分别表示的treatment中结果为Y的人数 、treatment中结果为N的人数。
 Fig2（b）展示的就是Two-model的一个结果例子。这种评估方式很有效，因为可以轻松的观察到treatment的整体的效果，可以通过选择不同的比例的人群进行施加action来获得最大利润。
   到现在为止，图片是无法提供一个衡量标准的，因此无法精确的比较两个模型效果。尽管如此，我们可以利用一下公式计算测试集上每一个分位的效果：

![f(t) = (\frac{Y_t^T}{N_t^T} - \frac{Y_t^C}{N_t^C} )(N_t^T+N_t^C) \tag{17}](https://math.jianshu.com/math?formula=f(t)%20%3D%20(%5Cfrac%7BY_t%5ET%7D%7BN_t%5ET%7D%20-%20%5Cfrac%7BY_t%5EC%7D%7BN_t%5EC%7D%20)(N_t%5ET%2BN_t%5EC)%20%5Ctag%7B17%7D)

这里t表示的按预测值进行排序的分位区间。
 Fig3(a)利用这种计算曲线的方式表示treatment的效果。图中随机(青色)的那条线表示对整个人群有个正向的结果(所以随着x轴的增加y值也增加)。曲线中的每一个点对应的是相应的uplif效果，这个值越高表示模型效果越好。uplfit曲线的连续性可以计算曲线下面的面积(就是之前提到的**AUUC**)，从而用来比较不同模型的效果。这种度量方式有点类似于AUC。在我们的案例中，Two-model始终要比其他的模型更好。在图中，如果曲线更像一个“钟”形状，表示模型效果更好；反之，更接近随机的曲线表示模型效果差。

![img](https:////upload-images.jianshu.io/upload_images/8374736-81b9b0cd7881e6f2.png?imageMogr2/auto-orient/strip|imageView2/2/w/1200)

uplift曲线



![img](https:////upload-images.jianshu.io/upload_images/8374736-5abfbb9445b1ac41.png?imageMogr2/auto-orient/strip|imageView2/2/w/1200)

Qini曲线



> 图中横坐标是人数，因为这里划分数据集是8：2划分的所以测试集人数为2k左右。

   在文献中，uplfit曲线是通过分别计算treatment组和control组的预测值按值排序计算分位值差异来定义的。这可能不是理想的方式，因为我们不能保证相同高分位值(都是前10% 或者都是前10-20%的分位区间)人群是相似的，但是这种计算方式在实践中的效果最好，并且是在随机实验情况下，这种方法是可行的。但是我们更倾向于我们的公式，因为它更接近Qini的原始定义：
 ![g(t)= Y_t^T- \frac{Y_t^C N_t^T}{N_t^C} \tag{18}](https://math.jianshu.com/math?formula=g(t)%3D%20Y_t%5ET-%20%5Cfrac%7BY_t%5EC%20N_t%5ET%7D%7BN_t%5EC%7D%20%5Ctag%7B18%7D)
 这里Qini系数表示的Qini曲线下的面积,如Fig3(b)所示，Qini曲线和Uplift曲线有些类似，因为：
 ![f(t) = \frac{g(t)(N_t^T+N_t^C)}{N_t^T} \tag{19}](https://math.jianshu.com/math?formula=f(t)%20%3D%20%5Cfrac%7Bg(t)(N_t%5ET%2BN_t%5EC)%7D%7BN_t%5ET%7D%20%5Ctag%7B19%7D)

## 4.3 基于![Y^*](https://math.jianshu.com/math?formula=Y%5E*)的度量

   在3.2章节中我们介绍了目标变量的转换形式![Y^*](https://math.jianshu.com/math?formula=Y%5E*)，因此很自然的想到在交叉验证中是否使用了预测值 ![\hat{\tau}](https://math.jianshu.com/math?formula=%5Chat%7B%5Ctau%7D)和真实值![Y^*](https://math.jianshu.com/math?formula=Y%5E*)的差异来表示的损失函数。这里定义:
 ![MSE(Y_i^{*},\hat{\tau}) = \sum_i^n \frac{1}{n}(Y_i^* -\hat{\tau_i} )^2 \tag{20}](https://math.jianshu.com/math?formula=MSE(Y_i%5E%7B*%7D%2C%5Chat%7B%5Ctau%7D)%20%3D%20%5Csum_i%5En%20%5Cfrac%7B1%7D%7Bn%7D(Y_i%5E*%20-%5Chat%7B%5Ctau_i%7D%20)%5E2%20%5Ctag%7B20%7D)
 做完近似：

![MSE(\tau_i,\hat{\tau}) = \sum_i^n \frac{1}{n}(\tau_i -\hat{\tau_i} )^2 \tag{21}](https://math.jianshu.com/math?formula=MSE(%5Ctau_i%2C%5Chat%7B%5Ctau%7D)%20%3D%20%5Csum_i%5En%20%5Cfrac%7B1%7D%7Bn%7D(%5Ctau_i%20-%5Chat%7B%5Ctau_i%7D%20)%5E2%20%5Ctag%7B21%7D)
 Athey and Imbens指出尽管可以我们可以从模拟数据中计算公式21中的数据，但是我们是无法从真实的观测数据中计算![\tau](https://math.jianshu.com/math?formula=%5Ctau)的。因为MSE无法计算，作者引入了一种只能在决策树种使用的估计量。这种指标强制了uplift使用特定的模型。我们的方法是不受限与机器学习的方法的。注意在前面的章节中，这个曲线也是基于局部的度量![\tau](https://math.jianshu.com/math?formula=%5Ctau) 。目的是我们可以用![Y^*](https://math.jianshu.com/math?formula=Y%5E*)来替代无法观测到的![\tau](https://math.jianshu.com/math?formula=%5Ctau)。
 改写公式21：

![img](https:////upload-images.jianshu.io/upload_images/8374736-53d3e67ad349d463.png?imageMogr2/auto-orient/strip|imageView2/2/w/1200)

公式22.png

。
 为了最小化MSE，我们可以可以忽视公式22中的第一项，因为它不依赖

![\hat{\tau_i}](https://math.jianshu.com/math?formula=%5Chat%7B%5Ctau_i%7D)

。在第二项中我们注意到

![\hat{\tau_i} \perp \tau_i |X_i](https://math.jianshu.com/math?formula=%5Chat%7B%5Ctau_i%7D%20%5Cperp%20%5Ctau_i%20%7CX_i)

  和 

![\hat{\tau_i} \perp Y_i^* |X_i](https://math.jianshu.com/math?formula=%5Chat%7B%5Ctau_i%7D%20%5Cperp%20Y_i%5E*%20%7CX_i)

，可以改写公式22中第二项。
 由于公式23中不依赖于我们的估计量，因此取极限时，最小化

![MSE(Y_i^*,\hat{\tau})](https://math.jianshu.com/math?formula=MSE(Y_i%5E*%2C%5Chat%7B%5Ctau%7D))

相等于最小化

![MSE(\tau_i,\hat{\tau_i})](https://math.jianshu.com/math?formula=MSE(%5Ctau_i%2C%5Chat%7B%5Ctau_i%7D))



> ![\hat{\tau_i} \perp \tau_i |X_i](https://math.jianshu.com/math?formula=%5Chat%7B%5Ctau_i%7D%20%5Cperp%20%5Ctau_i%20%7CX_i)  和 ![\hat{\tau_i} \perp Y_i^* |X_i](https://math.jianshu.com/math?formula=%5Chat%7B%5Ctau_i%7D%20%5Cperp%20Y_i%5E*%20%7CX_i)  ?????黑人问号？？？

![img](https:////upload-images.jianshu.io/upload_images/8374736-450339053b045c2f.png?imageMogr2/auto-orient/strip|imageView2/2/w/1200)

公式23.png

# 5 结论

   这篇介绍uplift模型的三种形式：1、Two-model ，2、类别转换的方式 ， 3、直接建模。指出Two-model形式简单，但是效果不好。类别转换的方式是依赖于完全随机的二分类实验。第三种方式是直接修改已有的学习器，主要讲了修改树学习器，提出不同的分裂标准。
   在模型评估方面，因为是无法观测到真正的uplfite增益值，所以一种方式是通过分别对treatment和control进行预测后的值进行排序，通过比较两个组的每个十分位的平均uplift来大概判断模型的效果，但这种无法精确的评估模型好坏。所以提出一种uplfit曲线和Qini曲线的方式，通过计算曲线下面的面积(类似于AUC)来表示模型的好坏。如果曲线更接近钟形，表示模型效果更好。

  