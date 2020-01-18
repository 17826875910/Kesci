
# 加载数据分析常用库
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_absolute_error,mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import csv
import warnings
import time
warnings.filterwarnings('ignore')

# 加载数据
file_train_path = r'F:\文件\PyCharmFiles\Kesci\真实业界数据的时间序列预测挑战\industry\industry_timeseries\timeseries_train_data\11.csv'
file_predict_path = r'F:\文件\PyCharmFiles\Kesci\真实业界数据的时间序列预测挑战\industry\industry_timeseries\timeseries_predict_data\11.csv'

data_train = pd.read_csv(file_train_path,names=['年','月','日','当日最高气温','当日最低气温','当日平均气温','当日平均湿度','输出'])
data_train['输出'] = data_train['输出'].shift(-1)
data= (data_train[0:-1][['当日最高气温','当日最低气温','当日平均气温','当日平均湿度','输出']]).values
# 定义参数
rnn_units = 10
input_size = 4
output_size = 1
lr = 0.0006
# 输入层、输出层权重、偏置
weights = {
            'in':tf.Variable(tf.random_normal([input_size,rnn_units])),
            'out':tf.Variable(tf.random_normal([rnn_units,output_size]))
          }
biases={
    'in':tf.Variable(tf.constant(0.1,shape=[rnn_units,])),
    'out':tf.Variable(tf.constant(0.1,shape=[output_size,]))
}
# 数据集划分和处理
def get_data(batch_size = 60, time_step = 20, train_begin = 0, train_end = 487):
    batch_index = []
    scaler_for_x = MinMaxScaler(feature_range=(0, 1))
    scaler_for_y = MinMaxScaler(feature_range=(0, 1))
    scaled_x_data = scaler_for_x.fit_transform(data[:, :-1])
    scaled_y_data = scaler_for_y.fit_transform(data[:, -1].reshape(-1,1))

    label_train = scaled_y_data[train_begin:train_end]
    label_test = scaled_y_data[train_end:]
    nommalized_train_data = scaled_x_data[train_begin:train_end]
    nommalized_test_data = scaled_x_data[train_end:]

    train_x,train_y=[],[]
    for i in range(len(nommalized_train_data)-time_step):
        if i % batch_size == 0:
            batch_index.append(i)
        x = nommalized_train_data[i:i+time_step, :4]
        y = label_train[i:i+time_step, :]
        train_x.append(x.tolist())
        train_y.append(y.tolist())
    batch_index.append((len(nommalized_train_data)-time_step))

    size = int(np.floor(len(nommalized_test_data)/time_step))
    test_x, test_y = [], []
    for i in range(size):
        x = nommalized_test_data[i*time_step:(i+1)*time_step, :4]
        y = label_test[i*time_step:(i+1)*time_step]
        test_x.append(x.tolist())
        test_y.append(y.tolist())

    # test_x.append((nommalized_test_data[i*time_step:(i+1)*time_step, :4]).tolist())
    # test_y.extend((label_test[(i+1)*time_step:]).tolist())

    return batch_index, train_x, train_y, test_x, test_y, scaler_for_y

# 定义神经网络变量
def lstm(X):
    batch_size = tf.shape(X)[0]
    time_step = tf.shape(X)[1]
    w_in = weights['in']
    b_in = biases['in']
    input = tf.reshape(X, [-1, input_size])
    input_rnn = tf.matmul(input, w_in)+b_in
    input_rnn=tf.reshape(input_rnn,[-1,time_step,rnn_units])
    cell=tf.contrib.rnn.BasicLSTMCell(rnn_units)

    init_state= cell.zero_state(batch_size,dtype=tf.float32)
    output_rnn,final_states = tf.nn.dynamic_rnn(cell, input_rnn, initial_state=init_state, dtype=tf.float32)
    # output_rnn是记录lstm每个输出节点的结果，final_states是最后一个cell的结果
    output=tf.reshape(output_rnn,[-1,rnn_units])  # 作为输入层的输入
    w_out = weights['out']
    b_out = biases['out']
    pred=tf.matmul(output,w_out)+b_out
    return pred,final_states


# 训练模型
def train_lstm(batch_size=80,time_step=15,train_begin=0,train_end=487):
    start_time = time.time()
    X = tf.placeholder(tf.float32,shape=[None,time_step,input_size])
    Y = tf.placeholder(tf.float32,shape=[None,time_step,output_size])
    batch_index,train_x,train_y,test_x,test_y,scaler_for_y = get_data(batch_size,time_step,train_begin,train_end)
    pred,_=lstm(X)
    # 损失函数
    loss=tf.reduce_mean(tf.square(tf.reshape(pred,[-1])-tf.reshape(Y,[-1])))
    train_op = tf.train.AdamOptimizer(lr).minimize(loss)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # 训练5000
        iter_time = 1000
        for i in range(iter_time):
            for step in range(len(batch_index)-1):
                mytest_x = train_x[batch_index[step]:batch_index[step+1]]
                mytest_y = train_y[batch_index[step]:batch_index[step+1]]
                _,loss_=sess.run([train_op,loss],feed_dict={X:train_x[batch_index[step]:batch_index[step+1]],Y:train_y[batch_index[step]:batch_index[step+1]]})
            if i % 100 == 0:
                print('iter:',i,'loss:',loss_)
        # predict
        test_predict=[]
        for step in range(len(test_x)):
            prob=sess.run(pred,feed_dict={X:[test_x[step]]})
            predict=prob.reshape((-1))
            test_predict.extend(predict)

        test_predict = scaler_for_y.inverse_transform(np.array(test_predict).reshape(-1, 1))
        test_y = scaler_for_y.inverse_transform(np.array(test_y).reshape(-1, 1))
        rmse = np.sqrt(mean_squared_error(test_predict, test_y))
        mae = mean_absolute_error(y_pred=test_predict, y_true=test_y)
        print('mae:', mae, 'rmse:', rmse)
        end_time = time.time()
        file_name = r'F:\文件\PyCharmFiles\Kesci\真实业界数据的时间序列预测挑战\result.csv'
        f = open(file_name, 'a', encoding='utf-8-sig')
        writer = csv.writer(f)
        writer.writerow([start_time,end_time,rnn_units,mae,rmse])
        f.close()
        plt.figure(figsize=(24, 8))
        plt.plot(data[:, -1])
        plt.plot([None for _ in range(487)] + [x for x in test_predict])
        plt.show()

    return test_predict


# test_predict = train_lstm(batch_size=80, time_step=15, train_begin=0, train_end=487)

def text(id = 1):
    a = id*-1;
    print(a)

if __name__ == '__main__':
    for i in range(5):
        print(i)
        text(i)
