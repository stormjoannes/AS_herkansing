value = [[1], [2], [3], [4],
         [5], [6], [7], [8],
         [9], [10], [11], [12],
         [13], [14], [15], [16]]

sizeHorizontal = 4
sizeVertical = 4
index = 0
maze = {}

for vertical in range(sizeVertical):
    for horizontal in range(sizeHorizontal):
        coord = (vertical, horizontal)
        maze[coord] = value[index]
        index += 1

print(maze[(3, 2)])
