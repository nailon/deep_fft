def int2bitvec(x, nbits):
    ret = []
    for i in range(nbits):
        ret.append(x & 1)
        x = x >> 1
    return ret


def bitvec2int(bitvec):
    ret = 0
    for i in range(len(bitvec)):
        ret = ret * 2 + bitvec[-1-i]
    return ret