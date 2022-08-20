import os
import random
import numpy as np
import matplotlib.image as image
import shutil


BROJ_CVETOVA = 3                                            # definisemo broj cvetova u slikama kao konstantu

# funkcija prosekRGB:
# prima "slika" kao putanju do slike,
# vraca vektor prosecne RGB vrednosti na celoj slici
def prosekRGB(slika):
    img=image.imread(slika)                                 # ucitavamo sliku kao matricu pomocu matplotlib

    brojPiksela=img.shape[0]*img.shape[1]                   # ucitavamo broj piksela slike da bi mogli da izracunamo proseke (w*h)
    
    prosecanRGB=np.sum(img, axis = (0, 1)) / brojPiksela    # sumiramo R, G, B vrednosti u slici zasebno i cuvamo ih u vektor,
                                                            # vektor delimo sa brojem piksela da nadjemo prosek
    print(f"Slika: {slika}\nProsecna RGB vrednost: {prosecanRGB}")

    return prosecanRGB

# funkcija init_centroids:
# prima "num_clusters" kao broj clustera,
# prima "boje" kao matricu prosecnih boja na N slika
# vraca: random izabranih num_clusters boja sa matrice
def init_centroids(num_clusters, boje):                     # boje je matrica koja sadrzi prosecne boje za svaku sliku
    centroids_init = np.zeros((num_clusters, 3))

    for i in range(num_clusters):                           # prolazimo kroz sve centroide 
        x = random.randrange(len(boje))                     # biramo random prosecan rgb
        centroids_init[i] = boje[x]                         # postavljamo kao centroid

    return centroids_init

# funkcija update_centroids:
# prima "centroids" kao pocetne centroide
# prima "boje" kao matricu prosecnih boja na slikama
def update_centroids(centroids, boje, max_iter=30, print_every=10):
    for iteracija in range(max_iter):
        pre = centroids.copy()                              # cuvamo centroide pre promene da vidimo da li se nista nije promenilo

        suma_centroida = np.zeros(centroids.shape)          # pamtimo sumu centroida u obliku centroida ispunjenu nulama
        br_centroida = np.zeros((len(centroids), 1))        # pamtimo broj centroida u obliku vektora sa len(centroids) redova
        # suma_centroida sabira sve R G B vrednosti po tome kom centroidu pripada
        # npr suma_centroida[0] ima sabrano sve boje koje pripadaju tom centroidu

        # br_centroida pamti broj centroida koji pripadaju svakom centroidu
        # npr br_centroida[0] ima broj clanova koji pripada centroidu 0

        for x in range(len(boje)):                          # prolazimo kroz sve prosecne boje u slikama
            r, g, b = boje[x]                               # izvlacimo r g b u varijable iz boje

            index_najblizeg = 0                             # kazemo da je clan 0 najblizi
            distanca_najblizeg = np.sqrt((centroids[0][0] - r) ** 2 + (centroids[0][1] - g) ** 2 + (centroids[0][2] - b) ** 2)
                                                            # racunamo razdaljinu pitagorinom teoremom

            for i in range(1, len(centroids)):              # prolazimo kroz sve centroide osim 0
                d = np.sqrt((centroids[i][0] - r) ** 2 + (centroids[i][1] - g) ** 2 + (centroids[i][2] - b) ** 2) #racunamo razdaljinu
                if distanca_najblizeg > d:                  # ako je razdaljina do centroida i manja
                    distanca_najblizeg = d                  # pamtimo taj centroid kao najblizi
                    index_najblizeg = i                     # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                
            suma_centroida[index_najblizeg] += [r, g, b]    # sabiramo r g b vrednosti na najblizem centroidu
            br_centroida[index_najblizeg] += 1              # dodajemo 1 na broj centroida najblizem centroidu

        for i in range(len(centroids)):                     # prolazimo kroz sve centroide
            if br_centroida[i] == 0:                        # da bi izbegli deljenje sa 0
                br_centroida[i] = 1                         # postavljamo 1 gde je bilo 0 na br_centroida

            # postavljamo novi centroid kao prosek boja koji pripadaju tom centroidu
            centroids[i] = np.true_divide(suma_centroida[i], br_centroida[i]) 

        if iteracija % print_every == 0:
            print("Iteracija " + str(iteracija))

        if np.all((centroids - pre) == 0):                  # proveravamo da li se neki centroid promenio
            print(f"Iteracija: {iteracija}, konacno: ")     # ako nije, prekidamo petlju i ispisujemo vrednosti
            print(centroids)
            break
    
    return centroids

# funkcija trening:
# izvrsava trening na slikama u direktorijumu ./slike
# vraca izracunate centroide
def trening():
    dir_path = "slike"
    lista_proseka = np.zeros((len(os.listdir(dir_path)), 3))
    j = 0
    for i in os.listdir(dir_path):
        rel_path = os.path.join(dir_path, i) # spajamo putanju sa imenom fajla da dobijemo relativnu putanju
        lista_proseka[j] = prosekRGB(rel_path)
        j += 1

    print(lista_proseka)
    centroids = init_centroids(BROJ_CVETOVA, lista_proseka)
    centroids = update_centroids(centroids, lista_proseka)
    return centroids

# funkcija napravi_foldere:
# brise foldere c0-cBROJ_CVETOVA i sve u njima ako postoje
# pravi nove foldere c0-cBROJ_CVETOVA
def napravi_foldere():
    for i in range(BROJ_CVETOVA):
        if os.path.exists("c" + str(i)):
            shutil.rmtree("c" + str(i))
        os.makedirs("c" + str(i))

# funkcija odredi_cvet:
# prima "slika" kao putanju do slike
# prima "centroids" kao centroide
# vraca: broj indexa najblizeg centroida prosecnom rgbu slike
def odredi_cvet(slika, centroids):
    r, g, b = prosekRGB(slika)                              # izvlacimo r, g, b iz proseka    

    index_najblizeg = 0                                     # pamtimo 0 kao najblizi i racunamo razdaljinu
    distanca_najblizeg = np.sqrt((centroids[0][0] - r) ** 2 + (centroids[0][1] - g) ** 2 + (centroids[0][2] - b) ** 2)

    for i in range(1, len(centroids)):                      # prolazimo kroz sve centroide i proveravamo da li je neki blizi
        d = np.sqrt((centroids[i][0] - r) ** 2 + (centroids[i][1] - g) ** 2 + (centroids[i][2] - b) ** 2)
        if d < distanca_najblizeg:
            index_najblizeg = i
            distanca_najblizeg = d
    
    return index_najblizeg

# funkcija rasporedi_cvetove:
# prima "centroids" kao centroide
# prolazi kroz sve slike
# i pravi kopiju u odgovarajucem folderu (c0-cBROJ_CVETOVA)
def rasporedi_cvetove(centroids):
    dir_path = "slike"
    j = 0
    for i in os.listdir(dir_path):                          # prolazimo kroz sve fajlove u folderu
        rel_path = os.path.join(dir_path, i)                # spajamo putanju sa imenom fajla da dobijemo relativnu putanju
        c = odredi_cvet(rel_path, centroids)                # pronalazimo kom centroidu pripada i kopiramo fajl u odgovarajuci folder
        shutil.copyfile(rel_path, os.path.join("c" + str(c), i))
        j += 1

def main():
    centroids = trening()
    napravi_foldere()
    rasporedi_cvetove(centroids)

main()