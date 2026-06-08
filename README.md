# Modelos

## Modal Set-Up
Primero para linkear tu computadora a Modal se debe hacer:
```
pip install modal
```
La version de pip debe ser al menos 26 para que funcione
```
python3 -m modal setup
```
## Corriendo Modal
Una vez que se tiene linkeado la computadora con la cuenta se debe correr:
```
modal serve example.py
```
Si modal no es un comando reconocido entonces se debe usar:
```
python -m modal serve example.py
```
## Accediendo a la pagina
Si se esta corriendo con serve se puede acceder en la ruta:
```
https://financiar186--example-get-started-main-dev.modal.run/?x=10
```
Donde x es el parametro de la funcion