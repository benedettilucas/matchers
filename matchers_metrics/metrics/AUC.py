import numpy as np
import torch

class AUCMetric:
    def __init__(self, thresholds, elements=None):
        # Garante que elements sempre seja uma lista
        self._elements = [] if elements is None else elements
        self.thresholds = thresholds if isinstance(thresholds, list) else [thresholds]

    def cal_error_auc(self, errors, thresholds):
        sort_idx = np.argsort(errors)
        errors = np.array(errors.copy())[sort_idx]
        recall = (np.arange(len(errors)) + 1) / len(errors)
        errors = np.r_[0.0, errors]
        recall = np.r_[0.0, recall]
        aucs = []
        for t in thresholds:
            last_index = np.searchsorted(errors, t)
            r = np.r_[recall[:last_index], recall[last_index - 1]]
            e = np.r_[errors[:last_index], t]
            aucs.append(np.round((np.trapz(r, x=e) / t), 4))
        return aucs

    def compute_mean_reprojection_error(self, H_est, H_gt, image_shape):
        """
        Calcula o erro médio de reprojeção dos quatro cantos da imagem.
        Args:
            H_est: np.ndarray (3x3) - homografia estimada
            H_gt: np.ndarray (3x3)  - homografia ground truth
            image_shape: (H, W)     - altura e largura da imagem
        Retorna:
            mean_error: float - erro médio de reprojeção (em pixels)
        """
        H, W = image_shape
        # 1. Cantos da imagem em coordenadas homogêneas
        corners = np.array([
            [0, 0, 1],
            [W, 0, 1],
            [0, H, 1],
            [W, H, 1]
        ]).T  # shape (3,4)

        # 2. Projeta os cantos com ambas homografias
        proj_est = H_est @ corners
        proj_gt  = H_gt  @ corners

        # 3. Converte para coordenadas cartesianas
        proj_est /= proj_est[2, :]
        proj_gt  /= proj_gt[2, :]

        # 4. Calcula erro euclidiano por canto
        errors = np.linalg.norm(proj_est[:2, :].T - proj_gt[:2, :].T, axis=1)

        # 5. Retorna erro médio (1 valor por par de imagens)
        return float(np.mean(errors))

    def update(self, tensor):
        if isinstance(tensor, float):
            tensor = torch.tensor([tensor])
        elif isinstance(tensor, list):
            tensor = torch.tensor(tensor)
        assert tensor.dim() == 1
        self._elements += tensor.cpu().numpy().tolist()

    def compute(self):
        if len(self._elements) == 0:
            return np.nan
        else:
            return self.cal_error_auc(self._elements, self.thresholds)