import pandas as pd
import numpy as np

mtcars = pd.read_csv('mtcars.csv')
mtcars.sort_values('mpg', inplace=True)
print(mtcars['car'].head(5))

mtcars.sort_values('cyl', inplace=True)
print(mtcars['car'].tail(3))

print(mtcars[mtcars.cyl==6].mpg.mean())

print(mtcars[(mtcars.cyl==4) & (mtcars.wt>=2.000) & (mtcars.wt<=2.200)].mpg.mean())

print(mtcars.am.value_counts())

am1_hp100 = mtcars[(mtcars.am==1) & (mtcars.hp>100)]
print(len(am1_hp100))

ukupno = mtcars['wt'].sum() * 1000
print(str(ukupno) + " kg")
