print("Unesite broj koji se nalazi izmedju 0.0 i 1.0")
broj = float(input())
if broj>=0.0 and broj<=1.0:
    if broj>=0.9:
        print("Ocjena: A")
    elif broj>=0.8:
        print("Ocjena: B")
    elif broj>=0.7:
        print("Ocjena: C")
    elif broj>=0.6:
        print("Ocjena: D")
    else:
        print("Ocjena: F")

else:    print("Krivi unos")

