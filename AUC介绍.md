---
marp: true
---
    
��� Python ����ʵ�ֵ��Ǽ��� AUC (Area Under the ROC Curve) �Ĺ��̡�ROC���ߣ�Receiver Operating Characteristic Curve�����������������ܵ�һ�ֹ��ߣ���AUC����ROC�����µ������ͨ����Ϊ�������������ܵ�һ����Ҫָ�ꡣ

1. `import numpy as np`: ���� NumPy �⣬���ڴ�����Ͷ�ά����;�������ݽṹ��

2. `import time`: ���� Python �� time ģ�飬��Ҫ����ʱ��Ļ�ȡ�͸�ʽ����

3. `def Calculation_AUC(MatrixAdjacency_Train,MatrixAdjacency_Test,Matrix_similarity,MaxNodeNum)`: ����һ����Ϊ "Calculation_AUC" �ĺ������������ĸ�������ѵ�����ڽӾ��󡢲��Ե��ڽӾ������ƶȾ���ͽڵ�����������

4. `AUC_TimeStart = time.time()`: ��¼������ʼִ�е�ʱ�䡣

5. `print('    Calculation AUC......')`: ��ӡһ����Ϣ����֪���ڿ�ʼ���� AUC��
---
6. `AUCnum = 672400`: ���� AUC �ļ��������

7. `Matrix_similarity = np.triu(Matrix_similarity - Matrix_similarity * MatrixAdjacency_Train)`: �����ƶȾ������Ϊѵ���ڽӾ�����δ���ӵĽڵ��Ӧ�����ƶȣ��������������ǲ��֡�

8. `Matrix_NoExist = np.ones(MaxNodeNum) - MatrixAdjacency_Train - MatrixAdjacency_Test - np.eye(MaxNodeNum)`: ���㲻�������ӵĽڵ�Եľ���

9. `Test = np.triu(MatrixAdjacency_Test)`: ��ȡ�����ڽӾ���������ǲ��֡�

10. `NoExist = np.triu(Matrix_NoExist)`: ��ȡ���������ӵĽڵ�Եľ���������ǲ��֡�
---
11-14. `Test_num = len(np.argwhere(Test == 1))` �� `NoExist_num = len(np.argwhere(NoExist == 1))`: �ֱ������Ծ���Ͳ��������ӵĽڵ�Ծ�����ֵΪ1��Ԫ��������

15-16. `Test_rd` �� `NoExist_rd`: ������0��`Test_num`��`NoExist_num`֮�����������б�

17-18. `TestPre` �� `NoExistPre`: �ֱ������Ծ���Ͳ��������ӵĽڵ�Ծ��������ƶȾ����Ԫ�س˻���

19-22. `TestIndex` �� `NoExistIndex`: �ֱ��ȡ���Ծ���Ͳ��������ӵĽڵ�Ծ�����ֵΪ1��Ԫ��������

---
23-24. `Test_Data` �� `NoExist_Data`: ���ݻ�ȡ������������`TestPre`��`NoExistPre`����ȡ��Ӧ��ֵ��

25-26. `Test_rd` �� `NoExist_rd`: ��`Test_Data`��`NoExist_Data`����ȡ��������ݡ�

27-35. ͨ������`Test_rd`��`NoExist_rd`���Ա�ÿ��Ԫ�ش�С������ AUC��

36. `print('    AUCָ��Ϊ��{0}'.format(auc))`: ��ӡ����õ��� AUC ֵ��

37. `AUC_TimeEnd = time.time()`: ��¼��������ִ�е�ʱ�䡣

38. `print('    AUCTime��{0} s'.format(AUC_TimeEnd - AUC_TimeStart))`: ��ӡ����ִ�������ѵ�ʱ�䡣

39. `return auc`: ���ؼ���õ��� AUC ֵ��
