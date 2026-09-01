"""
paper_csi_pipeline.py

Paper-based Wi-Fi CSI processing pipeline for NLoS fire/human detection.

Expected raw input:
    amplitude: ndarray, shape (N, 52)
    phase:     ndarray, shape (N, 52)
    labels:    ndarray, shape (N,)   # required for training / Fisher scoring

Class labels:
    0 = NFN  (No Fire, No Human)
    1 = NFH  (No Fire, Human)
    2 = FHN  (Fire, No Human)
    3 = FWH  (Fire, Human)

Pipeline implemented according to the paper:
1. Phase unwrapping + linear phase sanitization
2. Moving-average filtering, Lf = 10
3. Amplitude subcarrier scoring by SNR-like score:
       R_i^a = 20 log10(mu_i^a / s_i^a)
4. Phase subcarrier scoring by Fisher discriminant ratio
5. Select Ma amplitude + Mphase phase subcarriers
   Paper's reported optimum: Ma=8, Mphase=12
6. Min-max normalization using TRAINING-SET parameters only
7. Concatenate amplitude + phase features
8. MLP classification with ReLU hidden activation and multiclass output

Important:
- The paper specifies the MLP computation (ReLU + Softmax/cross-entropy)
  but does NOT give a concrete hidden-layer neuron count in the provided paper text.
  Therefore hidden_layer_sizes is exposed as a user parameter instead of pretending
  a specific architecture came from the paper.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


EPS = 1e-12
CLASS_NAMES = ["NFN", "NFH", "FHN", "FWH"]


def check_csi(amplitude: np.ndarray, phase: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Validate CSI input arrays."""
    amplitude = np.asarray(amplitude, dtype=np.float64)
    phase = np.asarray(phase, dtype=np.float64)

    if amplitude.ndim != 2 or phase.ndim != 2:
        raise ValueError("amplitude and phase must both be 2-D arrays: (N, subcarriers)")
    if amplitude.shape != phase.shape:
        raise ValueError(f"Shape mismatch: amplitude={amplitude.shape}, phase={phase.shape}")
    if amplitude.shape[1] != 52:
        raise ValueError(
            f"The paper uses 52 CSI subcarriers; received {amplitude.shape[1]} subcarriers."
        )
    if not np.all(np.isfinite(amplitude)) or not np.all(np.isfinite(phase)):
        raise ValueError("Input contains NaN or inf.")
    return amplitude, phase


def sanitize_phase(raw_phase: np.ndarray) -> np.ndarray:
    """
    Paper Eq. (6): phase sanitization.

    For each CSI packet:
    1) unwrap phase across the 52 subcarriers;
    2) fit/remove the linear component defined by the first and last subcarrier.

    Let eta_i be unwrapped phase and k_i the subcarrier index:
        eta_hat_i = eta_i -
                    [ (eta_52 - eta_1)/(k_52-k_1) * (k_i-k_1) + eta_1 ]

    This suppresses packet-dependent offsets / linear phase shifts while retaining
    environmental phase variation.
    """
    phase = np.asarray(raw_phase, dtype=np.float64)
    unwrapped = np.unwrap(phase, axis=1)

    n_sub = unwrapped.shape[1]
    k = np.arange(n_sub, dtype=np.float64)

    eta_first = unwrapped[:, [0]]
    eta_last = unwrapped[:, [-1]]

    slope = (eta_last - eta_first) / (k[-1] - k[0])
    fitted_linear = eta_first + slope * (k[None, :] - k[0])

    sanitized = unwrapped - fitted_linear
    return sanitized


def causal_moving_average(x: np.ndarray, window: int = 10) -> np.ndarray:
    """
    Paper Eq. (7)-(8): causal moving-average filter.

        x_bar(t) = 1/Lf * sum_{r=0}^{Lf-1} x(t-r)

    Only full windows are returned, so output length is N-window+1.
    """
    x = np.asarray(x, dtype=np.float64)
    if window <= 0:
        raise ValueError("window must be > 0")
    if x.shape[0] < window:
        raise ValueError(f"Need at least {window} packets, got {x.shape[0]}.")

    csum = np.vstack([np.zeros((1, x.shape[1])), np.cumsum(x, axis=0)])
    return (csum[window:] - csum[:-window]) / float(window)


def amplitude_snr_scores(filtered_amp: np.ndarray) -> np.ndarray:
    """
    Paper Eq. (9)-(11).

    mu_i = training-set mean of filtered amplitude subcarrier i
    s_i  = training-set standard deviation
    R_i  = 20*log10(mu_i / s_i)

    Higher score => stronger and more stable amplitude response.
    """
    mu = np.mean(filtered_amp, axis=0)
    # Paper formula uses 1/N rather than 1/(N-1), hence ddof=0.
    std = np.std(filtered_amp, axis=0, ddof=0)

    ratio = np.maximum(np.abs(mu), EPS) / np.maximum(std, EPS)
    return 20.0 * np.log10(ratio)


def phase_fisher_scores(filtered_phase: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """
    Paper Eq. (12)-(14): Fisher discriminant score for each phase subcarrier.

        numerator   = sum_c N_c * (mu_i,c - mu_i)^2
        denominator = sum_c sum_{t in class c} (eta_i(t) - mu_i,c)^2
        F_i         = numerator / denominator

    Higher score => better inter-class separability relative to intra-class variation.
    """
    X = np.asarray(filtered_phase, dtype=np.float64)
    y = np.asarray(labels)

    classes = np.unique(y)
    global_mean = np.mean(X, axis=0)

    between = np.zeros(X.shape[1], dtype=np.float64)
    within = np.zeros(X.shape[1], dtype=np.float64)

    for c in classes:
        Xc = X[y == c]
        if len(Xc) == 0:
            continue
        mu_c = np.mean(Xc, axis=0)
        between += len(Xc) * (mu_c - global_mean) ** 2
        within += np.sum((Xc - mu_c) ** 2, axis=0)

    return between / np.maximum(within, EPS)


@dataclass
class PaperCSIPreprocessor:
    """
    Training-time + inference-time preprocessing.

    Paper-reported settings:
        filter_window = 10
        Ma = 8
        Mphase = 12
    """
    filter_window: int = 10
    Ma: int = 8
    Mphase: int = 12

    amp_indices_: Optional[np.ndarray] = None
    phase_indices_: Optional[np.ndarray] = None
    amp_min_: Optional[np.ndarray] = None
    amp_max_: Optional[np.ndarray] = None
    phase_min_: Optional[np.ndarray] = None
    phase_max_: Optional[np.ndarray] = None

    def _filter(self, amplitude: np.ndarray, phase: np.ndarray):
        amplitude, phase = check_csi(amplitude, phase)
        phase_sanitized = sanitize_phase(phase)

        amp_f = causal_moving_average(amplitude, self.filter_window)
        phase_f = causal_moving_average(phase_sanitized, self.filter_window)
        return amp_f, phase_f

    def fit(self, amplitude: np.ndarray, phase: np.ndarray, labels: np.ndarray):
        """
        Fit subcarrier selection and min-max parameters from TRAINING DATA ONLY.
        """
        labels = np.asarray(labels)
        amplitude, phase = check_csi(amplitude, phase)

        if len(labels) != len(amplitude):
            raise ValueError("labels length must equal number of CSI packets.")

        amp_f, phase_f = self._filter(amplitude, phase)

        # Because filtering returns only complete causal windows, label each filtered
        # sample by the label at the current/end packet t.
        y_f = labels[self.filter_window - 1:]

        # 1) amplitude ranking
        amp_scores = amplitude_snr_scores(amp_f)
        amp_rank = np.argsort(amp_scores)[::-1]
        self.amp_indices_ = amp_rank[:self.Ma]

        # 2) phase ranking
        fisher_scores = phase_fisher_scores(phase_f, y_f)
        phase_rank = np.argsort(fisher_scores)[::-1]
        self.phase_indices_ = phase_rank[:self.Mphase]

        # 3) select
        amp_sel = amp_f[:, self.amp_indices_]
        phase_sel = phase_f[:, self.phase_indices_]

        # 4) min-max params: training set only
        self.amp_min_ = np.min(amp_sel, axis=0)
        self.amp_max_ = np.max(amp_sel, axis=0)
        self.phase_min_ = np.min(phase_sel, axis=0)
        self.phase_max_ = np.max(phase_sel, axis=0)

        return self

    def transform(self, amplitude: np.ndarray, phase: np.ndarray) -> np.ndarray:
        """
        Apply the fixed paper preprocessing steps to validation/test/online data.
        """
        if self.amp_indices_ is None or self.phase_indices_ is None:
            raise RuntimeError("Call fit() first.")

        amp_f, phase_f = self._filter(amplitude, phase)

        amp_sel = amp_f[:, self.amp_indices_]
        phase_sel = phase_f[:, self.phase_indices_]

        amp_norm = (amp_sel - self.amp_min_) / np.maximum(
            self.amp_max_ - self.amp_min_, EPS
        )
        phase_norm = (phase_sel - self.phase_min_) / np.maximum(
            self.phase_max_ - self.phase_min_, EPS
        )

        # Paper Eq. (22): concatenate amplitude and phase feature vectors.
        X = np.concatenate([amp_norm, phase_norm], axis=1)
        return X

    def fit_transform(self, amplitude, phase, labels):
        self.fit(amplitude, phase, labels)
        X = self.transform(amplitude, phase)
        y = np.asarray(labels)[self.filter_window - 1:]
        return X, y


class PaperCSIFireDetector:
    """
    Complete offline training + inference wrapper.

    The paper specifies:
      - MLP
      - ReLU hidden activation
      - Softmax class probabilities
      - cross-entropy training objective

    It does not state a concrete hidden-layer size in the supplied text, so this
    constructor requires/accepts hidden_layer_sizes as an implementation parameter.
    """

    def __init__(
        self,
        hidden_layer_sizes=(32,),
        filter_window=10,
        Ma=8,
        Mphase=12,
        random_state=42,
        max_iter=500,
    ):
        self.pre = PaperCSIPreprocessor(
            filter_window=filter_window,
            Ma=Ma,
            Mphase=Mphase,
        )

        self.model = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation="relu",
            solver="adam",
            max_iter=max_iter,
            random_state=random_state,
            early_stopping=True,
            validation_fraction=0.15,
        )

    def fit(self, amplitude, phase, labels):
        X_train, y_train = self.pre.fit_transform(amplitude, phase, labels)
        self.model.fit(X_train, y_train)
        return self

    def predict(self, amplitude, phase):
        X = self.pre.transform(amplitude, phase)
        return self.model.predict(X)

    def predict_proba(self, amplitude, phase):
        X = self.pre.transform(amplitude, phase)
        return self.model.predict_proba(X)

    def evaluate(self, amplitude, phase, labels):
        y_true = np.asarray(labels)[self.pre.filter_window - 1:]
        y_pred = self.predict(amplitude, phase)

        print("Accuracy:", accuracy_score(y_true, y_pred))
        print("\nClassification report:")
        print(
            classification_report(
                y_true,
                y_pred,
                labels=[0, 1, 2, 3],
                target_names=CLASS_NAMES,
                zero_division=0,
            )
        )
        print("Confusion matrix:")
        print(confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3]))

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            return pickle.load(f)


def load_csv_matrix(path: str) -> np.ndarray:
    """Load a CSV whose rows are packets and columns are 52 CSI subcarriers."""
    x = np.loadtxt(path, delimiter=",", dtype=np.float64)
    if x.ndim == 1:
        x = x[None, :]
    return x


def load_csv_labels(path: str) -> np.ndarray:
    """Load one integer label per row."""
    y = np.loadtxt(path, delimiter=",", dtype=np.int64)
    return np.asarray(y).reshape(-1)


if __name__ == "__main__":
    # ------------------------------------------------------------------
    # Example usage
    # ------------------------------------------------------------------
    #
    # File shapes:
    #   train_amplitude.csv : N_train x 52
    #   train_phase.csv     : N_train x 52
    #   train_labels.csv    : N_train rows, labels in {0,1,2,3}
    #
    #   test_amplitude.csv  : N_test x 52
    #   test_phase.csv      : N_test x 52
    #   test_labels.csv     : N_test rows
    #
    # Recommended paper settings:
    #   moving-average Lf = 10
    #   selected amplitude subcarriers = 8
    #   selected phase subcarriers = 12
    #   deployment sampling rate = 10 Hz
    #
    # NOTE:
    # The MLP hidden-layer size below is an implementation setting because the
    # provided paper text does not specify the exact neuron count.
    # ------------------------------------------------------------------

    train_amp = load_csv_matrix("train_amplitude.csv")
    train_phase = load_csv_matrix("train_phase.csv")
    train_y = load_csv_labels("train_labels.csv")

    test_amp = load_csv_matrix("test_amplitude.csv")
    test_phase = load_csv_matrix("test_phase.csv")
    test_y = load_csv_labels("test_labels.csv")

    detector = PaperCSIFireDetector(
        hidden_layer_sizes=(32,),  # implementation choice, NOT claimed as paper parameter
        filter_window=10,
        Ma=8,
        Mphase=12,
        max_iter=500,
    )

    detector.fit(train_amp, train_phase, train_y)

    print("Selected amplitude subcarrier indices:", detector.pre.amp_indices_)
    print("Selected phase subcarrier indices:", detector.pre.phase_indices_)

    detector.evaluate(test_amp, test_phase, test_y)

    # Save offline-trained pipeline for later inference.
    detector.save("csi_fire_detector.pkl")

    # Example prediction on new CSI sequence:
    # new_amp = load_csv_matrix("new_amplitude.csv")
    # new_phase = load_csv_matrix("new_phase.csv")
    # pred = detector.predict(new_amp, new_phase)
    # prob = detector.predict_proba(new_amp, new_phase)
    # print(pred)
    # print(prob)
