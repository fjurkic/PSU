import pandas as pd
import numpy as np
s1 = pd.Series(['crvenkapica', 'baka', 'majka', 'lovac', 'vuk'])
print(s1)
s2 = pd.Series(5., index=['a', 'b', 'c', 'd', 'e'], name = 'ime_objekta')
print(s2)
print(s2['b'])
s3 = pd.Series(np.random.randn(5))
print(s3)
print(s3[3])

data = {'country': ['Italy','Spain','Greece','France','Portugal'],
'population': [59, 47, 11, 68, 10],
'code': [39, 34, 30, 33, 351]}
countries = pd.DataFrame(data, columns=['country', 'population', 'code'])
print(countries)

mtcars = pd.read_csv('mtcars.csv')
print(len(mtcars))
print(mtcars)
print(mtcars.head(5))
print(mtcars.tail(3))
print(mtcars.info())
print(mtcars.describe())

print(mtcars['car'])
print(mtcars.cyl)
print(mtcars.cyl > 6)
print(mtcars[mtcars.cyl > 6])
print(mtcars[(mtcars.cyl == 4) & (mtcars.hp > 100)].car)
print(mtcars[['car','cyl']])
print(mtcars.cyl[2:4])

print(mtcars[5:12])
print(mtcars.mpg[3:5])
mtcars['jedinice'] = np.ones(len(mtcars))
mtcars['heavy'] = mtcars.wt > 4.5
print(mtcars[['car','heavy']])
print(mtcars.query('cyl == [4,6]').car)
print(mtcars.iloc[1:3, 5:10])
print(mtcars.iloc[:, 3:5])
print(mtcars.iloc[:, [0,4,7]])
print(mtcars.iloc[[1,29], :])


new_mtcars = mtcars.groupby('cyl')
print(new_mtcars.count())
print(new_mtcars.sum())
new_mtcars = mtcars.iloc[:,1:]
new_mtcars = new_mtcars.groupby('cyl')
print(new_mtcars.mean())