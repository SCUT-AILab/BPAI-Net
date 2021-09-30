import torch
import typing


class Accuracy(object):
    def __init__(self, rate, n_correct, n_total):
        self.rate = rate
        self.n_correct = n_correct
        self.n_total = n_total

    def __str__(self):
        return f"Accuracy={self.rate * 100:.4f}%({self.n_correct}/{self.n_total})"


class AccuracyMetric(object):
    def __init__(self, topk: typing.Iterable[int] = (1,)):
        self.topk = sorted(list(topk))
        self._last_accuracies = None
        self._accuracies = None
        self.reset()

    def reset(self, *args, **kwargs) -> None:
        self._accuracies = [
            Accuracy(rate=0.0, n_correct=0, n_total=0) for _ in self.topk]
        self.reset_last()

    def reset_last(self):
        self._last_accuracies = [
            Accuracy(rate=0.0, n_correct=0, n_total=0) for _ in self.topk]

    def update(self, targets, outputs) -> None:
        with torch.no_grad():
            maxk = max(self.topk)
            batch_size = targets.size(0)

            _, pred = outputs.topk(k=maxk, dim=1, largest=True, sorted=True)
            pred = pred.t()
            correct = pred.eq(targets.view(1, -1))

            for accuracy, last_accuracy, k in zip(self._accuracies, self._last_accuracies, self.topk):
                accuracy.n_total += batch_size

                correct_k = correct[:k].sum().item()
                accuracy.n_correct += correct_k
                accuracy.rate = accuracy.n_correct / accuracy.n_total

                last_accuracy.n_total = batch_size
                last_accuracy.n_correct = correct_k
                last_accuracy.rate = last_accuracy.n_correct / last_accuracy.n_total

    @property
    def value(self):
        return self

    def last_accuracy(self, i) -> Accuracy:
        return self._last_accuracies[self.topk.index(i)]

    def accuracy(self, i) -> Accuracy:
        return self._accuracies[self.topk.index(i)]


class AverageMetric(object):
    def __init__(self):
        self.n = 0
        self._value = 0.
        self.last = 0.

    def reset(self) -> None:
        self.n = 0
        self._value = 0.
        self.last = 0.

    def update(self, value) -> None:
        if torch.is_tensor(value):
            self.last = value.item()
        elif isinstance(value, (int, float)):
            self.last = value
        else:
            raise ValueError("The parameter 'value' should be int, float or pytorch scalar tensor, but found {}"
                             .format(type(value)))
        self.n += 1
        self._value += self.last

    @property
    def value(self) -> float:
        if self.n == 0:
            raise ValueError("The container is empty.")
        return self._value / self.n


class ConfusionMatrix(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.matrix = None
        self.reset()

    def reset(self):
        self.matrix = torch.zeros(size=(self.num_classes,)*2,
                                  dtype=torch.int64, device="cpu")

    def update(self, targets, predictions):
        predictions = torch.argmax(predictions, dim=1)
        targets, predictions = targets.flatten(), predictions.flatten()

        # targets * self.num_classes gets the start index of each class in a (class * class) length vector
        # actually is the 0 position of each line in matrix, then add predictions shifts the pred to responding column
        indices = targets * self.num_classes + predictions # 算出这些预测值在混淆矩阵还是一个向量时的位置，然后就可以用bincount来将对应位置置1，然后加到原矩阵就得到每类的预测数量
        m = torch.bincount(indices, minlength=self.num_classes **
                           2).reshape(self.num_classes, self.num_classes)

        self.matrix += m.to(device=self.matrix.device)

    def global_accuracy(self):
        m = self.matrix.float()
        return (m.diag().sum()/m.sum()+1e-15).item()

    def precision(self):
        m = self.matrix.float()
        return (m.diag() / (m.sum(dim=0, keepdim=False) + 1e-15)).tolist()

    def recall(self):
        m = self.matrix.float()
        return (m.diag() / (m.sum(dim=1, keepdim=False) + 1e-15)).tolist()
