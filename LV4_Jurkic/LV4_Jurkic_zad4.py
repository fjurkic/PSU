import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. ucitavanje ociscenih podataka
df = pd.read_csv('cars_processed.csv')
print("Informacije o datasetu:")
print(df.info())

# 1. Koliko mjerenja (automobila) je dostupno?
n_measurements = df.shape[0]
print(f"\n1. Broj automobila u datasetu: {n_measurements}")

# 2. Tip pojedinog stupca
print("\n2. Tipovi stupaca:")
print(df.dtypes)

# 3. Automobil s najvećom i najmanjom cijenom
max_price_car = df.loc[df['selling_price'].idxmax()]
min_price_car = df.loc[df['selling_price'].idxmin()]
print(f"\n3. Automobil s najvećom cijenom:\n{max_price_car}")
print(f"\nAutomobil s najmanjom cijenom:\n{min_price_car}")

# 4. Broj automobila proizvedenih 2012. godine
cars_2012 = df[df['year'] == 2012].shape[0]
print(f"\n4. Broj automobila proizvedenih 2012. godine: {cars_2012}")

# 5. Automobil koji je prešao najviše i najmanje kilometara
max_km_car = df.loc[df['km_driven'].idxmax()]
min_km_car = df.loc[df['km_driven'].idxmin()]
print(f"\n5. Automobil koji je prešao najviše kilometara:\n{max_km_car}")
print(f"\nAutomobil koji je prešao najmanje kilometara:\n{min_km_car}")

# 6. Najčešći broj sjedala
most_common_seats = df['seats'].mode()[0]
print(f"\n6. Najčešći broj sjedala: {most_common_seats}")

# 7. Prosječna kilometraža za Diesel i Petrol automobile
avg_km_diesel = df[df['fuel'] == 'Diesel']['km_driven'].mean()
avg_km_petrol = df[df['fuel'] == 'Petrol']['km_driven'].mean()
print(f"\n7. Prosječna kilometraža:")
print(f"Diesel: {avg_km_diesel:.2f} km")
print(f"Petrol: {avg_km_petrol:.2f} km")

# --- Vizualizacije ---
# Scatter plot km_driven vs selling_price po tipu goriva
sns.relplot(data=df, x='km_driven', y='selling_price', hue='fuel')
plt.title('Kilometraža vs Cijena po tipu goriva')
plt.show()

# Histogram selling_price
df['selling_price'].hist(bins=30)
plt.xlabel('Selling Price')
plt.ylabel('Frequency')
plt.title('Distribucija cijena automobila')
plt.show()

# Pretvaranje kategoričkih varijabli u 0 i 1 radi korelacije
cat_cols = df.select_dtypes(include='object').columns.tolist()
df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)

# Heatmap korelacija
plt.figure(figsize=(12,8))
sns.heatmap(df_encoded.corr(), annot=True, cmap='coolwarm', linewidths=1)
plt.title('Korelacija svih varijabli (numeričke i one-hot kodirane)')
plt.show()