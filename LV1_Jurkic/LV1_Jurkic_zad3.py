from statistics import mean

print("Unosite brojeve i kada zavrsite napisite Done")
brojevi = []
while True:
    unos = input()
    if unos.lower() == "done":
        break
    try:
        broj = float(unos)
        brojevi.append(broj)
    except ValueError:
        print("Greska")

srednja = mean(brojevi)
min = min(brojevi)
max = max(brojevi)
print("Srednja vrijednost: " + str(srednja))
print("Minimalna vrijednost: " + str(min))
print("Maksimalna vrijednost: " + str(max))