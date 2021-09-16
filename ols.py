import numpy as np # 矩阵
import numpy.linalg as la

class OLS(object):
    def __init__(self, X, y):
        self.X = np.array(X)
        self.y = np.array(y)

        self.betas = None

    """离合"""
    # 拟合
    def fit(self):
        # for OLS model
        self.betas = self.compute_betas(self.y, self.X)
        return self.betas

    def compute_betas(self, y, x):
        xT = x.T
        xtx = np.dot(xT, x)
        xtx_inv = la.inv(xtx)
        xtx_inv_xt = np.dot(xtx_inv, xT)
        betas = np.dot(xtx_inv_xt, y)
        return betas

    def predict(self, x):
        x = np.array(x)
        yhat = x.dot(self.betas)
        return np.array(yhat)



import pandas as pd
data = pd.read_excel(r'./data/Tokyo/Tokyomortality.xls',0) 

y = data['db2564']
X = data[["OCC_TEC", "OWNH", "POP65", "UNEMP"]]

X = np.array(X)


# # filename = "Tokyomortality.xls"

# filename = "data_singlevar.txt"
# X = []
# y = []
# with open(filename, 'r') as f:
#     for line in f.readlines():
#         xt, yt = [float(i) for i in line.split(',')]
#         X.append(xt)
#         y.append(yt)

# X = np.array(X).reshape((len(X), 1))
X = np.hstack([np.ones((len(X), 1)), X]) # 合并
y = np.array(y)

OLSmodel = OLS(X, y)
OLSmodel.fit()
y_pred = OLSmodel.predict(X)

print(OLSmodel.betas)
