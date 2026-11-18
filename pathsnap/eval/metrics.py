"""Shared evaluation metric interfaces. P2 owns implementation."""


def top1_accuracy(*args, **kwargs):
    raise NotImplementedError("P2: implement top1_accuracy")


def macro_f1(*args, **kwargs):
    raise NotImplementedError("P2: implement macro_f1")
