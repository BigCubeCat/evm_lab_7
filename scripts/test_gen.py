#!python
import random 


M = 10
N = 16

MIN_VALUE = -10
MAX_VALUE = 10

def generate_random_row(N: int) -> str:
    result = ""
    for i in range(N):
        value = random.randint(MIN_VALUE, MAX_VALUE)
        result += str(value) + " "
    return result + "\n"


def generate_one(N: int) -> str:
    res = f"{M}\n{N}\n"
    for i in range(N):
        res += "0 " * i + "1 " + "0 " * (N - i - 1) + "\n"
    return res


if __name__ == "__main__":
    with open("test.input", "w") as file:
        file.write(generate_one(N))

