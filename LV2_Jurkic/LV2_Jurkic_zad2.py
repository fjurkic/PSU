import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt(open("mtcars.csv", "rb"), usecols=(1,2,3,4,5,6),delimiter=",", skiprows=1)
mpg = data[:,0]
hp = data[:,3]
wt = data[:,5] * 100
plt.scatter(mpg, hp, color='blue', marker='o', s=wt, alpha=0.5)


plt.xlabel("mpg")
plt.ylabel("hp")
plt.title("MPG vs HP")

print(mpg.min())
print(mpg.max())
print(mpg.mean())

cyl = data[:,1]

mpg_6cyl = mpg[cyl == 6]

mpg_min_6 = np.min(mpg_6cyl)
mpg_max_6 = np.max(mpg_6cyl)
mpg_mean_6 = np.mean(mpg_6cyl)

print("6 cilindara:")
print("Min mpg:", mpg_min_6)
print("Max mpg:", mpg_max_6)
print("Srednji mpg:", mpg_mean_6)

plt.show()  
