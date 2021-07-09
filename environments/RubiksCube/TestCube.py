import timeit
from Cube import Cube

test_cube = Cube(3)

#test_cube.rotate_one_step(1,1,False)
test_cube.scramble(11)

print(test_cube.blocks)
print(test_cube.check_solved())
