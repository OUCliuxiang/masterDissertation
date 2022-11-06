import cfg
NUM_CLASSES = cfg.NUM_CLASSES

cates = [0] * NUM_CLASSES
with open("../data/val.txt", 'r') as f:
    for line in  f.readlines():
        line = line.strip().split()
        line[0] = line[0].split('/')
        if not cates[int(line[1])] == 0:
            continue
        cates[int(line[1])] = line[0][-2]
print(len(cates))
print(cates)