# 提交文件与测试方法说明

## 代码说明
关于实验过程中所修改过的python脚本文件以及其修改内容如下所示：
- layer.py: 该文件中定义了神经网络各种类型层对应的python类，Relu类表示仅有relu激活函数的层，Sigmoid类表示仅有sigmoid激活函数的层，Softmax类表示仅有softmax函数的层（仅进行归一化），Linear表示线性变换层；
- loss.py: 该文件中定义了欧式损失函数（对应类EuclideanLoss）和交叉熵损失函数（对应类CrossEntropyLoss）；
- run_mlp.py: 该文件是整个程序的主脚本，主要修改了的代码可以分类为四个部分，分别如下所示：
	* 从json描述文件中获取要训练的网络结构：该主脚本不再将网络结构及其参数直接写在脚本中，而是以json格式保存在额外的文件中，并在执行脚本的时候通过参数指定该文件；描述网络的具体格式可参考codes/models/中的json文件；
	* 将batch loss输出到指定文件：该主脚本会将训练过程中的网络的batch loss输出到指定文件中，该文件在执行脚本时通过参数指定；
	* 将模型精度输出到指定文件：该主脚本会将训练过程中的网络的模型精度输出到指定文件中，该文件在执行脚本时通过参数指定；
	* 进行数据增强：通过旋转，平移，加噪声等变换，获得新的数据，数据增强只会在指定了-da参数的时候才执行；
- solve_net.py: 该文件是用于训练测试网络的脚本，主要修改部分为增加了往指定文件中输出模型精度和batch loss的代码；

## 文件说明
关于所提交的各个文件功能说明如下：
- codes：该目录中包括的实验所需代码，网络模型描述文件，以及进行测试的脚本，其中各个文件或子目录功能如下：
	* 以py为拓展名的文件： 训练与测试MLP的python脚本；
	* test.sh: 用于测试上文中提及的所有模型的脚本；
	* models: 保存用于描述要训练的网络结构的json格式文件，其中各个文件的所描述的网络如下所示：
		+ structure.json: 默认的存放网络结构的文件；
		+ softmax.json: 一个简单的softmax回归；
		+ test.json：单、双隐藏层（使用Sgimoid和Relu作为激活函数）的MLP；
		+ test_bs.json：用于探讨batch size对模型精度影响所使用的网络；
		+ test_wd.json：用于探讨weight decay对模型精度影响所使用的网络；
		+ test_mm.json：用于探讨momentum对模型精度影响所使用的网络；
		+ test_lr.json：用于探讨learning rate对模型精度影响所使用的网络；
		+ test_da.json：上文数据增强部分使用的三隐藏层MLP；
	* resutls: 默认用于保存训练结果的目录, 目录下不包含任何文件；
- README：代码与测试说明文件；
- report.pdf: 实验报告;

## 测试说明：
运行程序进行网络的训练与测试的主脚本是run_mlp.py；执行格式如下所示；
python run_mlp.py [file of network structures][file to store network accuracy][file to store batch loss][-da]
各个参数的含义如下所示：
- 参数[file of network structures]: 指定描述要训练的网络的结构的文件，为json格式, 缺省则该文件为models/structure.json;
- 参数[file to store network accuracy]: 指定要储存训练过程中每一个epoch中，网络在test数据集上的精度，缺省则为results/accuracy.txt;
- 参数[file to store batch loss]: 指定网络训练过程中网络在batch上的损失函数数值，该输出与输出在屏幕上的loss一致，缺省则默认输出到results/loss.txt；
- 参数[-da]：用于指定是否使用数据增强的参数，如果包含了-da参数则开启数据增强，默认不使用数据增强；

比如说，要训练的网络结构储存在models/test.json文件中，然后精度和损失函数值分别要保存到results/testacc.txt, results/testloss.txt中，并且开启数据增强，则训练和测试网络的命令应为：
python run_mlp.py models/test.json results/testacc.txt results/testloss.txt -da

要重现实验报告中各个模型的训练结果，在codes目录下提供了进行测试的脚本test.sh，只需执行sh test.sh则可进行测试，其中各个命令的功能如下：
- python run_mlp.py models/test.json results/test_acc.txt results/test_loss.txt：测试单、双层神经网络在不同激活函数选择下的性能；
- python run_mlp.py models/test_lr.json results/acc_lr_new.txt results/loss_lr_new.txt：测试learning rate对模型精度影响；
- python run_mlp.py models/test_wd.json results/acc_wd_new.txt results/loss_wd_new.txt：测试weight decay对模型精度影响；
- python run_mlp.py models/test_mm.json results/acc_mm_new.txt results/loss_mm_new.txt：测试momentum对模型精度影响；
- python run_mlp.py models/test_bs.json results/acc_bs_new_.txt results/loss_bs_new.txt：测试batch size对模型精度影响；
- python run_mlp.py models/test_da.json results/acc_da_new.txt results/loss_da_new.txt -da：测试使用了数据增强的模型精度；
