import numpy as np

from block6_eval.thresholding import predict_with_threshold, threshold_by_quota

def test_quota_0_2_selects_about_one_of_five():
    y = np.array([0.1, 0.2, 0.9, 0.8, 0.7])
    t = threshold_by_quota(y, quota=0.2)
    pred = predict_with_threshold(y, threshold=t)

    assert int(pred.sum()) == 1