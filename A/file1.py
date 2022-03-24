from B.Bee import B
# import B.Cee. iC
from C.Cee import C

print("This is the name attribute", __name__, __package__)
print("This is file 1")
def func():
    print("Hello, I am in file 1 and inside func")
    return 1
B.B1()