import sys
import numpy as np

class LinearRegression:
    def __init__(self, base_functions: np.ndarray, learning_rate: float, reg_coefficient: float):
        self.epochs_cnt = 50
        self.cost_funcs = [0] * self.epochs_cnt
        self.base_funcs = base_functions
        self.weights = np.random.randn(np.size(self.base_funcs) + 1)
        self.lr = learning_rate
        self.rc = reg_coefficient


    def predict(self, inputs: np.ndarray) -> np.ndarray:
        plan_matrix = self._calc_plan_matrix(inputs)
        return self._calc_predictions(plan_matrix)

    
    def train(self, inputs: np.ndarray, targets: np.ndarray, train_normal_eq: bool) -> None:
        if train_normal_eq:
            self._train_normal_eq(inputs, targets)
        else:
            self._train_grad(inputs, targets)

    
    def _train_normal_eq(self, inputs: np.ndarray, targets: np.ndarray) -> None:
        plan_matrix = self._calc_plan_matrix(inputs)
        pseudoinverse_plan_matrix = self._calc_pseudoinverse_matrix(plan_matrix)
        self.weights = pseudoinverse_plan_matrix @ targets


    def _train_grad(self, inputs: np.ndarray, targets: np.ndarray) -> None:
        plan_matrix = self._calc_plan_matrix(inputs)
        for e in range(self.epochs_cnt):
            grad = self._calc_grad(plan_matrix, targets)
            self.weights = self.weights - self.lr * grad

            if e % 10 == 0:
                predictions = self._calc_predictions(plan_matrix)
                cost_func = self._calc_cost_func(predictions, targets)
                self.cost_funcs[e] = cost_func


    def _calc_predictions(self, plan_matrix: np.ndarray) -> np.ndarray:
        return plan_matrix @ self.weights.T


    def _calc_grad(self, plan_matrix: np.ndarray, targets: np.ndarray) -> np.ndarray:
        result = (2 / self.n) * plan_matrix.T @ (plan_matrix @ self.weights - targets) + self.rc * self.weights
        return result


    def _calc_cost_func(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        result = 1 / self.n * np.sum(np.square(targets - predictions)) + self.rc * self.weights.T @ self.weights
        return result


    def _calc_plan_matrix(self, inputs: np.ndarray) -> np.ndarray:
        self.n = np.size(inputs)
        self.m = np.size(self.base_funcs)
        result = np.ones((self.n, self.m+1)) # матрица, все значения которой равны 1
        for i in range(self.m):
            result[:, i+1] = np.vectorize(self.base_funcs[i](inputs)) # [:, i+1] - весь (i+2)-ой столбец матрицы
        return result

    
    def _calc_pseudoinverse_matrix(self, matrix: np.ndarray) -> np.ndarray:
        eps = sys.float_info.epsilon
        u, s, vh = np.linalg.svd(matrix)
        s_s = np.zeros_like(matrix)
        s_sh = np.where(s > eps * max(self.n, self.m+1) * np.max(s), s / (s**2 + self.rc), 0)
        s_s[:len(s_sh), :len(s_sh)] = np.diag(s_sh)
        return vh.T @ s_s.T @ u.T
