import modal

image = modal.Image.debian_slim().pip_install("fastapi[standard]")
app = modal.App("example-get-started")


@app.function()
def square(x):
    print("This code is running on a remote worker!")
    return x**2

@app.function()
def square_local(x):
    print("This code is running locally!")
    return x**2


@app.function(image=image)
@modal.fastapi_endpoint()
def main(x: int):
    """
    Esta funcion corre el ambiente remoto pero llama a la funcion square_local que corre local
    Hay varias manera de correr funciones en modal:
    f.local() -> Corre la funcion localmente
    f.remote() -> Se corre una maquina de modal
    f.map () -> Corre la funcion en paralelo en modal
    """
    return {"square": square_local.local(x)}