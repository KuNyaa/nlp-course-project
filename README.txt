����Ҫ��
    python 3.5+
    tensorflow 1.4������
    java

ʹ��˵����
    main.py Ϊ��ģ��������
    �������¿�ѡ����
    --data_path
        Ĭ��Ϊ"./data/turk/"
        ��ʾ���ݼ������ļ���
        ����src-train.txt, src-val.txt, src-test.txt�ֱ�����Ȼ��������
        ����targ-train.txt, targ-val.txt, targ-test.txt�ֱ��Ŷ�Ӧ��������ʽ
        ÿһ��Ϊһ��sample
    --load
        Ĭ��ΪFalse
        ��ʾ�Ƿ�����м�����ز���(True/False)
        ��ΪTrue����ģ�ͻ�����ѵ�����֣�ֱ�ӳ��Դӵ�ǰģ�ͳ�������Ӧ��Ŀ¼�¶�ȡģ�Ͳ���
    --inference
        Ĭ��ΪFalse
        True��ʾ�ɵ�Դ��Ȼ���Խ���������ʽ���ɣ�False��ʾ�Բ��Լ����в���
        ��ΪTrue������Ҫ��data_pathָ����Ŀ¼�´��src-infer.txt�ļ����ļ���ʽͬ�ϣ���ʾ��Ҫ���ɵ�sample
        ��Ӧ��������������result-infer.txt��
    --vocab_freq_threshold
        Ĭ��Ϊ3
        ģ�ͳ���������ʾ���ʵ���С���ִ���
    --epochs
        Ĭ��Ϊ90
        ģ�ͳ�������ѵ���������ݼ��Ĵ���
    --batch_size
        Ĭ��Ϊ128
        ģ�ͳ����������δ�С
    --num_layers
        Ĭ��Ϊ3
        ģ�ͳ�������RNN�ѵ��Ĳ���
    --learning_rate
        Ĭ��Ϊ0.001
        ģ�ͳ���������ʼѧϰ��
    --embedding_size
        Ĭ��Ϊ256
        ģ�ͳ���������Ƕ��������С
    --hidden_size
        Ĭ��Ϊ512
        ģ�ͳ�������GRU��Ԫ��������Ĵ�С
    --dropout_keep_prob
        Ĭ��Ϊ0.6
        ģ�ͳ�������dropout��������

    ������
        ��ǰ��https://github.com/KuNyaa/nlp-course-project.git
        (���������ַ��û���ҵ�����ʹ�ø�����http://cloud.fudan.edu.cn/shareFolder/528460001/SmnA1khizDng)
        �������е�checkpoints�ļ���
        ���ҽ�ģ�Ͳ�������Ϊcheckpoints�ļ����ڶ�Ӧ�ı�ʶ��ģ�Ͳ���(����������Ĭ�ϲ�����ͬ)

        �����Ҫ��֤���ݼ��ϵ���ȷ�ʣ������������
            python main.py --load=True

        �����Ҫ����������ʽ(��Ҫ�ṩsrc-infer.txt)�������������
            python main.py --load=True --inference=True

