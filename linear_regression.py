import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets,linear_model
     
def get_data(file_name):
    data = pd.read_csv(file_name)
    x_paras = []
    y_paras = []
    for single_square_feet,single_price in zip(data['square_feet'],data['price']):
        x_paras.append([float(single_square_feet)])
        y_paras.append([float(single_price)])
    return x_paras,y_paras

x,y = get_data('/Users/terryjohnpaul/Desktop/input_data.csv')
print (x)
print (y)

def linear_model_main(x_paras,y_paras,predict_value):
    reg = linear_model.LinearRegression()
    reg.fit(x_paras,y_paras)
    predict_outcome = reg.predict(predict_value)
    predictions ={}
    predictions['intercept'] = reg.intercept_
    predictions['coefficient'] = reg.coef_
    predictions['predicted_value'] = predict_outcome
    return predictions

x,y = get_data('/Users/terryjohnpaul/Desktop/input_data.csv')
predict_value = 500
result = linear_model_main(x,y,predict_value)
print ("Intercept value " , result['intercept'])
print ("coefficient" , result['coefficient'])
print ("Predicted value: ",result['predicted_value'])


def show_linear_line(x_paras,y_paras):
    regr = linear_model.LinearRegression()
    regr.fit(x_paras, y_paras)
    plt.scatter(x_paras,y_paras,color='green')
    plt.plot(x_paras,regr.predict(x_paras),color='red',linewidth=5)
    plt.xticks(())
    plt.yticks(())
    plt.show()

show_linear_line(x,y)
