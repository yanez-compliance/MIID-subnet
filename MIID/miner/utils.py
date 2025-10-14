# _mat_str = lambda mat: "+" * 34 + "\n" + ("\n" + "+" * 34 + "\n").join("\n".join("|" + " ".join(f"{element:3d}" for element in subrow) + " |" for subrow in row) for row in mat) + "\n" + "+" * 34
def _mat_str84x(mat):
    res = ""
    res += "+" * 34 + "\n"
    for row in mat:
        for subrow in row:
            row_str = "|" + " ".join(f"{element:3d}" for element in subrow) + " |"
            res += row_str + "\n"
        res += "+" * 34 + "\n"
    return res

def _mat_str84(mat):
    res = ""
    res += "+" * 34 + "\n"
    for subrow in mat:
        row_str = "|" + " ".join(f"{element:3d}" for element in subrow) + " |"
        res += row_str + "\n"
    res += "+" * 34 + "\n"
    return res

# import random
# mat=[[[random.randint(0, 100) for _ in range(8)] for _ in range (4)] for _ in range(3)]
# print(_mat_str(mat))