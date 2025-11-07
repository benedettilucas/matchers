import numpy as np
import cv2

def random_homography(h, w, max_angle=20, max_scale=0.15, max_shift=0.05):
    # Coordenadas normalizadas dos cantos
    pts1 = np.float32([[0,0], [w,0], [w,h], [0,h]])

    # Parâmetros aleatórios
    angle = np.deg2rad(np.random.uniform(-max_angle, max_angle))
    scale = 1.0 + np.random.uniform(-max_scale, max_scale)
    tx = np.random.uniform(-max_shift, max_shift) * w
    ty = np.random.uniform(-max_shift, max_shift) * h

    # Matriz de rotação + escala
    R = np.array([
        [np.cos(angle)*scale, -np.sin(angle)*scale, tx],
        [np.sin(angle)*scale,  np.cos(angle)*scale, ty],
        [0, 0, 1]
    ])

    # Pequeno deslocamento perspectivo (efeito 3D planar)
    persp = np.array([
        [1, 0, np.random.uniform(-1e-4, 1e-4)],
        [0, 1, np.random.uniform(-1e-4, 1e-4)],
        [np.random.uniform(-1e-7, 1e-7), np.random.uniform(-1e-7, 1e-7), 1]
    ])

    H = persp @ R
    pts2 = cv2.perspectiveTransform(pts1[None, :, :], H)[0]
    return H, pts1, pts2