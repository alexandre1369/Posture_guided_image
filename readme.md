# Posture guided image

## Comment run le code avec les réseaux entrainés

Dans le fichier `DanceDemo.py`, modifier la variable `GEN_TYPE` pour :

    - 1 : Nearest 
    - 3 : GenVanilla (avec le meilleur réseau) 
    - 4 : GenGan  (avec le meilleur réseau) 

## Comment entrainer les réseaux 

Dans le fichier `GenGAN.py` :
```python
    if True:    # train or load
        # Train
        gen = GenGAN(targetVideoSke, False)
        gen.train(75) #5) #200)
    else:
        gen = GenGAN(targetVideoSke, loadFromFile=True) 
```

Modifier la valeur en `True` pour entrainer le modèle et `False` pour utiliser le meilleur. Attention, en entrainant, vous écraserez le meilleur réseau.


Dans le fichier `GenVanillaNN.py`  :
```python
    train = 1 #False
    .
    .
    .
    if train:
        # Train
        gen = GenVanillaNN(targetVideoSke, loadFromFile=False)
        gen.train(n_epoch)
    else:
        gen = GenVanillaNN(targetVideoSke, loadFromFile=True)    # load from file
```

Modifier la valeur de train en `1` pour entrainer le modèle et `0` pour utiliser le meilleur. Attention, en entrainant, vous écraserez le meilleur réseau.

## Ce que nous avons fait

`GenVanillaNN`

`GenGAN` : L'idée du GAN est de réaliser un `discriminator` qui pourra détecter quand une image généré est vrai ou pas. Dans notre `discriminator` nous réalisons une suite de couche Conv2d en montant jusqu'a 128 puis redescendon progressivement pour finir avec une couche 16 1 puis une fonction d'activation Sigmoid. 

Le train entraine sur plusieurs époques. Il alterne entre l’entraînement d’un discriminateur, qui apprend à différencier les images réelles des images générées, et un générateur, qui essaie de créer des images suffisamment réalistes pour tromper le discriminateur. À chaque époque, le code calcule les pertes (erreurs) des deux réseaux.
