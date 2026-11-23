"""실습 6·7용 소용돌이 데이터와 그림.

원고의 ToyData 호출 방식에 맞춰 새로 만든 교육용 모듈입니다.
원래 배포 파일이나 책 그림의 정확한 데이터를 복원한 것은 아닙니다.
기본값: 클래스마다 학습 100개, 테스트 100개, 난수 시드 0.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch


class ToyData:
    def __init__(self, num_classes=3, points_per_class=100, seed=0):
        self.num_classes = num_classes
        rng = np.random.default_rng(seed)

        def make_data():
            x, y = [], []
            for label in range(num_classes):
                r = rng.uniform(0.02, 1.0, points_per_class)
                theta = (2 * np.pi * r + 2 * np.pi * label / num_classes
                         + rng.normal(0, 0.12, points_per_class))
                x.append(np.column_stack((r * np.sin(theta), r * np.cos(theta))))
                y.append(np.full(points_per_class, label, dtype=np.int64))
            x = np.concatenate(x).astype(np.float32)
            y = np.concatenate(y)
            order = rng.permutation(len(y))
            return x[order], y[order]

        # 같은 분포에서 따로 생성하여 학습 데이터와 테스트 데이터를 분리합니다.
        self.xtrain, self.ytrain = make_data()
        self.xtest, self.ytest = make_data()

    def load_data(self):
        """NumPy 배열 xtrain, ytrain, xtest, ytest를 이 순서로 반환합니다."""
        return (self.xtrain.copy(), self.ytrain.copy(),
                self.xtest.copy(), self.ytest.copy())

    def plot_spiraldata(self):
        """학습 데이터를 표시하고 (fig, ax)를 반환합니다."""
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(self.xtrain[:, 0], self.xtrain[:, 1], c=self.ytrain,
                   cmap='Spectral', vmin=0, vmax=self.num_classes - 1, s=20)
        ax.set(xlim=(-1.1, 1.1), ylim=(-1.1, 1.1),
               title='Spiral Dataset (colored by class)', aspect='equal')
        return fig, ax

    def visualize_model(self, model, entropy=False):
        """model은 (M, 2) float32 텐서를 받아 (M, 클래스 수) 확률을 반환합니다."""
        grid = np.linspace(-1.1, 1.1, 200, dtype=np.float32)
        xx, yy = np.meshgrid(grid, grid)
        points = np.column_stack((xx.ravel(), yy.ravel()))
        with torch.no_grad():
            probabilities = model(torch.from_numpy(points))
        if isinstance(probabilities, torch.Tensor):
            probabilities = probabilities.detach().cpu().numpy()
        probabilities = np.asarray(probabilities)

        fig, ax = plt.subplots(figsize=(6, 6))
        if entropy:
            p = np.clip(probabilities, 1e-12, 1.0)
            values = -(probabilities * np.log(p)).sum(axis=1)
            im = ax.contourf(xx, yy, values.reshape(xx.shape), levels=20,
                             cmap='viridis')
            fig.colorbar(im, ax=ax, label='Entropy')
        else:
            labels = probabilities.argmax(axis=1).reshape(xx.shape)
            ax.contourf(xx, yy, labels,
                        levels=np.arange(self.num_classes + 1) - 0.5,
                        cmap='Spectral', vmin=0, vmax=self.num_classes - 1,
                        alpha=0.45)
        ax.scatter(self.xtrain[:, 0], self.xtrain[:, 1], c=self.ytrain,
                   cmap='Spectral', vmin=0, vmax=self.num_classes - 1,
                   s=20, edgecolors='k', linewidths=0.25)
        ax.set(xlim=(-1.1, 1.1), ylim=(-1.1, 1.1),
               title='Model classification' if not entropy else 'Model entropy',
               aspect='equal')
        return fig, ax
