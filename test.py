import numpy as np

def thetatoq(thetas, lamb):

    q = 4 * np.pi / lamb * np.sin(np.radians(thetas) / 2)
    print(f"✓ 计算q值: {q} (theta: {thetas}, lambda: {lamb})")

    return q

