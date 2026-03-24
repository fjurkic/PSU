import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

mtcars = pd.read_csv('mtcars.csv')

mpg_by_cyl = mtcars.groupby('cyl')['mpg'].mean()
mpg_by_cyl.plot(kind='bar')
plt.title('Prosječna potrošnja automobila po broju cilindara')
plt.xlabel('Broj cilindara')
plt.ylabel('Potrošnja (mpg)')
plt.show()

weights = [mtcars[mtcars.cyl == 4]['wt'],
           mtcars[mtcars.cyl == 6]['wt'],
           mtcars[mtcars.cyl == 8]['wt']]
plt.boxplot(weights, labels=['4 cilindra', '6 cilindra', '8 cilindra'])
plt.title('Distribucija težine automobila po broju cilindara')
plt.ylabel('Težina (1000 lbs)')
plt.show()

manual = mtcars[mtcars.am == 1]['mpg']
automatic = mtcars[mtcars.am == 0]['mpg']
plt.boxplot([manual, automatic], labels=['Ručni', 'Automatski'])
plt.title('Potrošnja automobila po tipu mjenjača')
plt.ylabel('Potrošnja (mpg)')
plt.show()


manual_cars = mtcars[mtcars.am == 1]
automatic_cars = mtcars[mtcars.am == 0]

plt.scatter(manual_cars.hp, manual_cars.qsec, color='blue', label='Ručni')
plt.scatter(automatic_cars.hp, automatic_cars.qsec, color='red', label='Automatski')
plt.title('Ubrzanje vs snaga automobila po tipu mjenjača')
plt.xlabel('Snaga (hp)')
plt.ylabel('Ubrzanje')
plt.legend()
plt.show()