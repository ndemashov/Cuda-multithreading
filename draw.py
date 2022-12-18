import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from celluloid import Camera

def euclid(x1, x2, y1, y2):
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

# vectors = [x1, x2, y, phi1, phi2, Ax, Ay, Bx, By]
def draw_balon_dynamic(path: str, save_path: str): # path = '../../smth.csv', save_path = '../../smth.mp4'
    with open(path, 'r'):
        vectors = pd.read_csv(path).to_numpy()
    
    fig = plt.figure()
    camera = Camera(fig)
    for X in vectors:
        radius1 = euclid(X[0], X[5], X[2], X[6])
        radius2 = euclid(X[1], X[7], X[2], X[8])

        plt.xlim(-1, 1)
        plt.ylim(-1, 1)
        alpha1 = (((3*np.pi)/2)-X[3])
        alpha2 = (X[4]-(np.pi/2))
        angles1 = np.linspace(alpha1, alpha1+X[3], 20)
        angles2 = np.linspace(-(np.pi/2), alpha2, 20)

        xs1 = []
        ys1 = []
        xs2 = []
        ys2 = []

        for j in range(len(angles1)):
            xs1.append(radius1 * np.cos(angles1[j]) + X[0])
            ys1.append(radius1 * np.sin(angles1[j]) + X[2])
            xs2.append(radius2 * np.cos(angles2[j]) + X[1])
            ys2.append(radius2 * np.sin(angles2[j]) + X[2])


        plt.plot(xs1, ys1, color='black', lw=4)
        plt.plot(xs2, ys2, color='black', lw=4)
        plt.scatter([X[0], X[1]], [X[2], X[2]], c='blue', s=10)
        plt.scatter(X[5], X[6], c='r', s=100)
        plt.scatter(X[7], X[8], c='r', s=100)
        camera.snap()

    plt.xlabel('Ось x', fontsize=10, color='black')
    plt.ylabel('Ось y', fontsize=10, color='black')
    animation = camera.animate()
    plt.show()
    animation.save(save_path)