# Smart Music Guardian
![image](https://user-images.githubusercontent.com/67227342/236187078-21b0929f-a751-45cb-93f3-a4afcb4e98a3.png)

## 安装环境需求
我们的主要运行环境为Python 3.8、pytorch=1.10.2、udatoolkit=11.3、scikit-learn=1.2.1

## 数据集
（1）[Free ST Chinese Mandarin Corpus]: http://www.openslr.org/resources/38/ST-CMDS-20170001_1-OS.tar.gz  
   
（2）[Twitter US Airline Sentiment]: https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment   


（3）[CIFAR100]: http://www.cs.toronto.edu/~kriz/cifar.html  


## 文件结构
|- ./  

|-- codes  

|--- CAM  

|--- jaro-winkler-distance-master  

|--- voice  

|--- models  

|--- ...  

|-- server  


其中codes部分包括后端代码，server包含前端代码。codes中，CAM文件夹包括CAM模型的相关代码，jaro-winkler-distance-master文件夹包括Jaro-Winkler similarity实现代码，voice文件夹包括语音识别的相关代码，剩余部分则是LSTM的训练及应用代码，models文件夹中包含所需多个模型，其余部分则是LSTM即SVM的相关代码。

### 测试结果
<img width="1184" alt="image" src="https://user-images.githubusercontent.com/67227342/236192799-d041defa-c95e-450f-ada8-6cfc9f8a3779.png">
<img width="1186" alt="image" src="https://user-images.githubusercontent.com/67227342/236192884-dd41bf2f-d2e6-4015-acab-08ca3b483379.png">


