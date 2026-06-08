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
    return {"square": square_local.local(x)}