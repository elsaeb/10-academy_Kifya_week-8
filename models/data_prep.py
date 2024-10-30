import pickle
from sklearn.model_selection import train_test_split
data1 = '../Data/fraud_data_processed.pkl'

with open(data1, 'rb') as file:
    fraud_data = pickle.load(file)

fraud_data=fraud_data.astype('float64')

data2 = '../Data/credit_data_processed.pkl'

with open(data2, 'rb') as file:
    credit = pickle.load(file)

#Separate features (X) and target (y)
X = fraud_data.drop(['class'], axis=1) 
y = fraud_data['class']
X_ = credit.drop(['Class'], axis=1) 
y_ = credit['Class']

X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(X_, y_, test_size=0.3, random_state=42, stratify=y_)
