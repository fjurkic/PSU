print("Unesite ime tekstualne datoteke:")
ime_datoteke = input()
putanja = "C:\\Users\\Jurkic\\Desktop\\" + ime_datoteke
fhand = open(ime_datoteke)

zbroj = 0
brojac = 0

for line in fhand:
    if line.startswith("X-DSPAM-Confidence:"):
        dijelovi = line.split(":")
        broj = float(dijelovi[1])
        zbroj = zbroj + broj
        brojac = brojac + 1

prosjek = zbroj / brojac
print("Prosječna vrijednost:", prosjek)