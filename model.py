#importing libraries
import pandas as pd
import pickle


if __name__ =="__main__":
    df = pd.read_csv("salary_predict_dataset.csv")
    df['test_score'].fillna(df['test_score'].mean(), inplace=True)
    X = df.iloc[:, :3]
    #Converting words to integer values
    def convert_to_int(w):
        word = {'one':1, 'two':2, 'three':3, 'four':4, 'five':5, 'six':6, 'seven':7, 'eight':8,
                'nine':9, 'ten':10, 'eleven':11, 'twelve':12, 'zero':0, 
                '0': 0,
                'fifteen':15,
                'thirteen':13}
        return word[w]
    
    X['experience'] = X['experience'].apply(lambda x : convert_to_int(x))
    y = df.iloc[:, -1]
    
    
    from sklearn.linear_model import LinearRegression
    lr = LinearRegression()
    lr.fit(X,y)
    
     #saving pkl file using pickle module
    pickle.dump(lr,open('model.pkl','wb'))