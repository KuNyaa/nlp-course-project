环境要求：
    python 3.5+
    tensorflow 1.4及以上
    java

使用说明：
    main.py 为本模型主程序
    共有如下可选命令
    --data_path
        默认为"./data/turk/"
        表示数据集所在文件夹
        其中src-train.txt, src-val.txt, src-test.txt分别存放自然语言描述
        其中targ-train.txt, targ-val.txt, targ-test.txt分别存放对应的正则表达式
        每一行为一个sample
    --load
        默认为False
        表示是否从已有检查点加载参数(True/False)
        若为True，则模型会跳过训练部分，直接尝试从当前模型超参数对应的目录下读取模型参数
    --inference
        默认为False
        True表示由单源自然语言进行正则表达式生成，False表示对测试集进行测试
        若为True，则需要在data_path指定的目录下存放src-infer.txt文件，文件格式同上，表示需要生成的sample
        对应的输出结果保存在result-infer.txt中
    --vocab_freq_threshold
        默认为3
        模型超参数，表示单词的最小出现次数
    --epochs
        默认为90
        模型超参数，训练整个数据集的次数
    --batch_size
        默认为128
        模型超参数，批次大小
    --num_layers
        默认为3
        模型超参数，RNN堆叠的层数
    --learning_rate
        默认为0.001
        模型超参数，初始学习率
    --embedding_size
        默认为256
        模型超参数，词嵌入向量大小
    --hidden_size
        默认为512
        模型超参数，GRU单元输出向量的大小
    --dropout_keep_prob
        默认为0.6
        模型超参数，dropout保留概率

    样例：
        请前往https://github.com/KuNyaa/nlp-course-project.git
        (如果上述地址内没有找到，请使用复旦云http://cloud.fudan.edu.cn/shareFolder/528460001/SmnA1khizDng)
        下载其中的checkpoints文件夹
        并且将模型参数设置为checkpoints文件夹内对应的标识的模型参数(其参数与程序默认参数相同)

        如果想要验证数据集上的正确率，输入以下命令：
            python main.py --load=True

        如果想要生成正则表达式(需要提供src-infer.txt)，输入以下命令：
            python main.py --load=True --inference=True

