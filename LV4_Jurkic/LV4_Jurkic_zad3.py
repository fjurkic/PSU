import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures

def non_func(x):
    y = 1.6345 - 0.6235*np.cos(0.6067*x) - 1.3501*np.sin(0.6067*x) \
        - 1.1622 * np.cos(2*x*0.6067) - 0.9443*np.sin(2*x*0.6067)
    return y

def add_noise(y):
    np.random.seed(14)
    varNoise = np.max(y) - np.min(y)
    y_noisy = y + 0.1*varNoise*np.random.normal(0,1,len(y))
    return y_noisy


x = np.linspace(1,10,50)
y_true = non_func(x)
y_measured = add_noise(y_true)

x = x[:, np.newaxis]
y_measured = y_measured[:, np.newaxis]


degrees = [2, 6, 15]
MSEtrain = []
MSEtest = []

plt.figure(figsize=(10,6))
plt.plot(x, y_true, 'k-', label='True function')

for d in degrees:
    poly = PolynomialFeatures(degree=d)
    X_poly = poly.fit_transform(x)

    np.random.seed(12)
    indices = np.random.permutation(len(X_poly))
    train_idx = indices[:int(0.7*len(X_poly))]
    test_idx = indices[int(0.7*len(X_poly)):]

    X_train = X_poly[train_idx]
    y_train = y_measured[train_idx]

    X_test = X_poly[test_idx]
    y_test = y_measured[test_idx]

   
    model = lm.LinearRegression()
    model.fit(X_train, y_train)

    
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    MSEtrain.append(mean_squared_error(y_train, y_train_pred))
    MSEtest.append(mean_squared_error(y_test, y_test_pred))


    plt.plot(x, model.predict(X_poly), label=f'Model degree={d}')

plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Comparison of polynomial models with true function')
plt.show()

print("MSE on training data:", MSEtrain)
print("MSE on test data:", MSEtest)


sample_sizes = [20, 50, 100] 
for n_samples in sample_sizes:
    x_sample = np.linspace(1,10,n_samples)[:, np.newaxis]
    y_sample = add_noise(non_func(x_sample.flatten()))[:, np.newaxis]

    poly = PolynomialFeatures(degree=15)
    X_poly_sample = poly.fit_transform(x_sample)

    np.random.seed(12)
    indices = np.random.permutation(len(X_poly_sample))
    train_idx = indices[:int(0.7*len(X_poly_sample))]
    test_idx = indices[int(0.7*len(X_poly_sample)):]

    X_train = X_poly_sample[train_idx]
    y_train = y_sample[train_idx]

    X_test = X_poly_sample[test_idx]
    y_test = y_sample[test_idx]

    model = lm.LinearRegression()
    model.fit(X_train, y_train)

    y_test_pred = model.predict(X_test)
    mse_test = mean_squared_error(y_test, y_test_pred)

    print(f"Number of samples: {n_samples}, Test MSE: {mse_test:.4f}")