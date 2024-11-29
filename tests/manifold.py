from typing import TypeVar


# Define a generic class with inline parameters
class LinearMap[M, N]:
    def transform(self, input: M) -> N: ...


# Define a TypeVar bound to the class
TLinearMap = TypeVar("TLinearMap", bound=LinearMap)


# Function that works with any subclass of LinearMap
def process_map(map: TLinearMap) -> None:
    print(f"Processing a map: {map}")
