import os
import random
import numpy as np
import matplotlib.image as image
import shutil

BROJ_CVETOVA = 3

def prosekRGB(slika):
    img=image.imread(slika)

    brojPiksela=img.shape[0]*img.shape[1]
    
    prosecanRGB=np.sum(img, axis = (0, 1)) / brojPiksela
    print(f"Slika: {slika}\nProsecna RGB vrednost: {prosecanRGB}")

    return prosecanRGB

def init_centroids(num_clusters, boje): # boje je matrica koja sadrzi prosecne boje za svaku sliku
    centroids_init = np.zeros((num_clusters, 3))

    for i in range(num_clusters): # prolazimo kroz sve centroide 
        x = random.randrange(len(boje)) # biramo random prosecan rgb
        centroids_init[i] = boje[x] # postavljamo kao centroid

    return centroids_init

def update_centroids(centroids, boje, max_iter=30, print_every=10):
    new_centroids = centroids

    for iteracija in range(max_iter):
        pre = new_centroids.copy()

        suma_centroida = np.zeros(new_centroids.shape)
        br_centroida = np.zeros((len(new_centroids), 1))

        for x in range(len(boje)):
            r, g, b = boje[x]

            najblizi = new_centroids[0].copy()
            index_najblizeg = 0
            distanca_najblizeg = np.sqrt((new_centroids[0][0] - r) ** 2 + (new_centroids[0][1] - g) ** 2 + (new_centroids[0][2] - b) ** 2)

            for i in range(1, len(new_centroids)):
                d = np.sqrt((new_centroids[i][0] - r) ** 2 + (new_centroids[i][1] - g) ** 2 + (new_centroids[i][2] - b) ** 2)
                if distanca_najblizeg > d:
                    distanca_najblizeg = d
                    index_najblizeg = i
                    najblizi = new_centroids[i]
            
            suma_centroida[index_najblizeg] += [r, g, b]
            br_centroida[index_najblizeg] += 1

        for i in range(len(new_centroids)):
            if br_centroida[i] == 0:
                br_centroida[i] = 1
            new_centroids[i] = np.true_divide(suma_centroida[i], br_centroida[i])

        if iteracija % print_every == 0:
            print("Iteracija " + str(iteracija))

        if np.all((new_centroids - pre) == 0):
            print(f"Iteracija: {iteracija}, konacno: ")
            print(new_centroids)
            break
    
    return new_centroids

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

def napravi_foldere():
    for i in range(BROJ_CVETOVA):
        if os.path.exists("c" + str(i)):
            shutil.rmtree("c" + str(i))
        os.makedirs("c" + str(i))

def odredi_cvet(slika, centroids):
    r, g, b = prosekRGB(slika)

    index_najblizeg = 0
    distanca_najblizeg = np.sqrt((centroids[0][0] - r) ** 2 + (centroids[0][1] - g) ** 2 + (centroids[0][2] - b) ** 2)

    for i in range(1, len(centroids)):
        d = np.sqrt((centroids[i][0] - r) ** 2 + (centroids[i][1] - g) ** 2 + (centroids[i][2] - b) ** 2)
        if d < distanca_najblizeg:
            index_najblizeg = i
            distanca_najblizeg = d
    
    return index_najblizeg


def rasporedi_cvetove(centroids):
    dir_path = "slike"
    j = 0
    for i in os.listdir(dir_path):
        rel_path = os.path.join(dir_path, i) # spajamo putanju sa imenom fajla da dobijemo relativnu putanju
        c = odredi_cvet(rel_path, centroids)
        shutil.copyfile(rel_path, os.path.join("c" + str(c), i))
        j += 1

def main():
    centroids = trening()
    napravi_foldere()
    rasporedi_cvetove(centroids)

main()