import torch

from torchmetrics import Metric


class FalseDiscoveryRate(Metric):
    def __init__(self, num_classes: int, **kwargs):
        """Кастомная метрика False Discovery Rate

        FP - False Positive
        TP - True Positive
        FDR = FP / (FP + TP)

        Args:
            **kwargs:
                num_classes (int): количество классов
        """
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.add_state(
            "tp_per_class",
            default=torch.zeros(self.num_classes).to(torch.int32),
            dist_reduce_fx="sum"
        )
        self.add_state(
            "fp_per_class",
            default=torch.zeros(self.num_classes).to(torch.int32),
            dist_reduce_fx="sum"
        )

    def update(self, preds, target):
        """Обновляет состояния метрики

        Args:
            preds (torch.Tensor): Тензор предсказаний модели.
            target (torch.Tensor): Тензор истинных меток.
        """
        if preds.shape[0] != target.shape[0]:
            raise ValueError("preds and target must have the same shape")

        pred_classes = torch.argmax(preds, dim=1)

        true_positives = (pred_classes == target).to(torch.int32)
        false_positives = (pred_classes != target).to(torch.int32)

        tp_per_class = torch.bincount(
            target[true_positives == 1],
            minlength=self.num_classes
        )
        fp_per_class = torch.bincount(
            pred_classes[false_positives == 1],
            minlength=self.num_classes
        )

        self.tp_per_class += tp_per_class
        self.fp_per_class += fp_per_class

    def compute(self):
        """
        Вычисляет метрику False Discovery Rate (FDR)
        как среднее арифметическое FDR по всем классам.

        FDR - это метрика, описывающая долю
        ложноположительных из общего числа положительных примеров.

        Returns:
            torch.Tensor: FDR, средняя метрика по всем классам
        """
        positive_count = self.fp_per_class + self.fp_per_class

        # Отладка для случая, когда FP + TP == 0
        zero_div_mask = positive_count == 0
        positive_count[zero_div_mask] = 1

        fdr = self.fp_per_class.float() / positive_count.float()
        fdr[zero_div_mask] = 0.0

        return fdr.mean()
