def add(*x):
    return sum(x)


print(add())

def configure(**kwargs):
    for key, value in kwargs.items():
        print(f"{key}: {value}")

configure(name="Alice", age=30, city="New York")