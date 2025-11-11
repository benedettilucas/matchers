import numpy as np
import torch
import cv2

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

    def compute_dlt_error(self, kpts0, kpts1, H_gt, image_shape):
        """
        Estima H_est via DLT e calcula erro médio de reprojeção (em pixels)
        entre H_est e H_gt.
        """
        # Converte keypoints para formato (N, 2)
        kpts0 = np.asarray(kpts0)
        kpts1 = np.asarray(kpts1)

        if len(kpts0) < 4 or len(kpts1) < 4:
            return np.nan  # DLT precisa de pelo menos 4 pares de pontos

        # Estima homografia via DLT
        H_est, mask = cv2.findHomography(kpts0, kpts1, method=0)  # 0 = DLT puro
        if H_est is None or np.linalg.matrix_rank(H_est) < 3:
            return np.nan

        # Calcula erro médio de reprojeção entre H_est e H_gt
        return self.compute_mean_reprojection_error(H_est, H_gt, image_shape)


    def compute_dlt(self, pairs, image_shape):
        """
        Calcula AUC baseado em DLT a partir de pares de keypoints e homografias GT.

        Args:
            pairs: lista de tuplas [(kpts0, kpts1, H_gt), ...]
            image_shape: tupla (H, W)

        Returns:
            dict com erros médios e AUCs para thresholds especificados
        """
        reprojection_errors = []
        for (kpts0, kpts1, H_gt) in pairs:
            err = self.compute_dlt_error(kpts0, kpts1, H_gt, image_shape)
            if not np.isnan(err):
                reprojection_errors.append(err)

        if len(reprojection_errors) == 0:
            return {f"AUC-DLT@{t}px": np.nan for t in self.thresholds}

        aucs = self.cal_error_auc(reprojection_errors, self.thresholds)
        return {f"AUC-DLT@{t}px": aucs[i].tolist() for i, t in enumerate(self.thresholds)}

    def update(self, tensor):
        if isinstance(tensor, float):
            tensor = torch.tensor([tensor])
        elif isinstance(tensor, list):
            tensor = torch.tensor(tensor)
        assert tensor.dim() == 1
        self._elements += tensor.cpu().numpy().tolist()

    def compute_ransac(self):
            if len(self._elements) == 0:
                return {f"AUC-RANSAC@{t}px": np.nan for t in self.thresholds}

            aucs = self.cal_error_auc(self._elements, self.thresholds)
            return {f"AUC-RANSAC@{t}px": aucs[i].tolist() for i, t in enumerate(self.thresholds)}

