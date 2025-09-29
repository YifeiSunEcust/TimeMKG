import numpy as np


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(true - pred))


def MSE(pred, true):
    return np.mean((true - pred) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((true - pred) / true))


def MSPE(pred, true):
    return np.mean(np.square((true - pred) / true))

def smape(preds, trues):
    return 100 * np.mean(2 * np.abs(preds - trues) / (np.abs(preds) + np.abs(trues) + 1e-8))

def mase(preds, trues, insample):
    """
    preds: [N, T, D]
    trues: [N, T, D]
    insample: [N, T_insample, D]  # 训练集真实值
    """
    n = insample.shape[1]
    d = np.mean(np.abs(insample[:, 1:, :] - insample[:, :-1, :]))
    errors = np.mean(np.abs(preds - trues))
    return errors / (d + 1e-8)

def owa(preds, trues, insample, naive_preds):
    """
    OWA = 0.5*(MASE_model/MASE_naive + sMAPE_model/sMAPE_naive)
    naive_preds: [N, T, D]  # naive预测（如前一时刻值）
    """
    mase_model = mase(preds, trues, insample)
    mase_naive = mase(naive_preds, trues, insample)
    smape_model = smape(preds, trues)
    smape_naive = smape(naive_preds, trues)
    return 0.5 * (mase_model / (mase_naive + 1e-8) + smape_model / (smape_naive + 1e-8))

def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)

    return mae, mse, rmse, mape, mspe
