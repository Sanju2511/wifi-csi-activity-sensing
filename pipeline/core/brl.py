"""Contextual Bayesian RL agent.

State: 12 PCA components + bias + one-hot previous decision (17 values).
Actions: IDLE, STAND, WALK. Reward: 1 if correct, else 0.
Per-action Gaussian belief over linear weights, Thompson sampling in training,
and abstention (UNCERTAIN) when the 90% credible intervals overlap.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ACTIONS = ["IDLE", "STAND", "WALK"]
ACTION_INDEX = {a: i for i, a in enumerate(ACTIONS)}
N_ACTIONS = len(ACTIONS)
START_TOKEN = N_ACTIONS       # extra previous-action slot: "no history yet"
N_PREV_SLOTS = N_ACTIONS + 1


class ContinuousBayesianRL:
    def __init__(self, n_pca=12, seed=42, n_passes=40,
                 ci=0.90, min_evidence=1.0, lambda_reg=1.0,
                 memory=True, temp=4.0, **kwargs):
        self.n_pca = n_pca
        self.seed = seed
        self.n_passes = n_passes
        self.ci = ci
        self.min_evidence = min_evidence
        self.lambda_reg = lambda_reg
        self.memory = memory
        self.temp = temp

        self.scaler = None
        self.pca = None
        self.feature_cols = None

        # context = PCA values + bias + one-hot previous action
        self.dim = None
        self.A_mat = None      # [A_0, A_1, A_2] precision matrices
        self.b_vec = None      # [b_0, b_1, b_2] reward accumulators

        # used by the out-of-distribution check
        self.manifold_mean = None
        self.manifold_cov_inv = None
        self.ll_floor = None
        self.ll_train_median = None
        self.ll_train_p01 = None

    def _project(self, Xr):
        """Scale the raw features and project them to PCA space."""
        X = self.scaler.transform(Xr)
        return self.pca.transform(X) if self.pca is not None else X

    def _build_context(self, z, prev):
        """Build the 17-value state vector."""
        ctx = np.zeros(self.dim, dtype=float)
        ctx[:self.n_pca] = z
        ctx[self.n_pca] = 1.0   # bias term
        if self.memory:
            slot = int(prev) if prev < N_PREV_SLOTS else START_TOKEN
            ctx[self.n_pca + 1 + slot] = 1.0
        else:
            ctx[self.n_pca + 1 + START_TOKEN] = 1.0
        return ctx

    def _episodes(self, df):
        """One episode per activity recording, in time order."""
        for label, g in df.groupby("label"):
            g = g.sort_values("window_index")
            Z = self._project(g[self.feature_cols].to_numpy())
            truth = g["label"].map(ACTION_INDEX).to_numpy()
            yield label, Z, truth

    def fit(self, train_df, feature_cols, verbose=True):
        self.feature_cols = list(feature_cols)
        Xr = train_df[self.feature_cols].to_numpy()
        self.scaler = StandardScaler().fit(Xr)
        Xs = self.scaler.transform(Xr)

        if self.n_pca and self.n_pca < Xs.shape[1]:
            self.pca = PCA(n_components=self.n_pca, random_state=self.seed).fit(Xs)
            Z_train = self.pca.transform(Xs)
            if verbose:
                print(f"  PCA: {Xr.shape[1]} features -> {self.n_pca} components "
                      f"({self.pca.explained_variance_ratio_.sum()*100:.1f}% of variance)")
        else:
            self.n_pca = Xs.shape[1]
            self.pca = None
            Z_train = Xs

        self.dim = self.n_pca + 1 + N_PREV_SLOTS

        # prior: A = lambda*I, b = 0 for every action
        self.A_mat = [self.lambda_reg * np.eye(self.dim, dtype=float) for _ in range(N_ACTIONS)]
        self.b_vec = [np.zeros(self.dim, dtype=float) for _ in range(N_ACTIONS)]

        self._fit_plausibility(Z_train)

        rng = np.random.default_rng(self.seed)
        episodes = list(self._episodes(train_df))

        for _pass in range(self.n_passes):
            for _label, Z, truth in episodes:
                prev = START_TOKEN
                for t in range(len(truth)):
                    ctx = self._build_context(Z[t], prev)

                    # Thompson sampling: sample weights from each belief, pick the best
                    sampled_scores = np.zeros(N_ACTIONS, dtype=float)
                    for a in range(N_ACTIONS):
                        cov = np.linalg.pinv(self.A_mat[a])
                        mu = cov @ self.b_vec[a]
                        theta_sample = rng.multivariate_normal(mu, cov)
                        sampled_scores[a] = float(ctx @ theta_sample)

                    a_chosen = int(np.argmax(sampled_scores))
                    correct = (a_chosen == truth[t])
                    reward = 1.0 if correct else 0.0

                    # bandit feedback: only the chosen action is updated
                    self.A_mat[a_chosen] += np.outer(ctx, ctx)
                    self.b_vec[a_chosen] += reward * ctx
                    prev = a_chosen

        if verbose:
            print(f"  fitted: Continuous Contextual Bayesian RL ({self.dim} context dimensions)")
        return self

    def _fit_plausibility(self, Z, pct=1.0):
        self.manifold_mean = np.mean(Z, axis=0)
        cov = np.cov(Z, rowvar=False) + 1e-4 * np.eye(Z.shape[1])
        self.manifold_cov_inv = np.linalg.pinv(cov)

        diff = Z - self.manifold_mean
        d2 = np.sum(diff @ self.manifold_cov_inv * diff, axis=1)
        ll = -0.5 * d2

        med = float(np.median(ll))
        p01 = float(np.percentile(ll, pct))
        self.ll_floor = p01 - 3.0 * (med - p01)
        self.ll_train_median = med
        self.ll_train_p01 = p01

    def plausibility(self, feature_vector):
        """Returns (log_density, is_plausible)."""
        if self.ll_floor is None:
            return float("nan"), True
        z = self._project(np.asarray(feature_vector, dtype=float).reshape(1, -1))[0]
        diff = z - self.manifold_mean
        d2 = float(diff @ self.manifold_cov_inv @ diff)
        ll = -0.5 * d2
        return ll, ll >= self.ll_floor

    def decide(self, input_vector, prev):
        """Returns (reported or None, best guess, prob, lower, upper, evidence)."""
        if len(input_vector) == len(self.feature_cols) if self.feature_cols else len(input_vector) > self.n_pca:
            z = self._project(np.asarray(input_vector, dtype=float).reshape(1, -1))[0]
        else:
            z = np.asarray(input_vector[:self.n_pca], dtype=float)

        ctx = self._build_context(z, prev)

        means = np.zeros(N_ACTIONS, dtype=float)
        stds = np.zeros(N_ACTIONS, dtype=float)
        evidences = np.zeros(N_ACTIONS, dtype=float)

        for a in range(N_ACTIONS):
            cov = np.linalg.pinv(self.A_mat[a])
            mu = cov @ self.b_vec[a]
            means[a] = float(ctx @ mu)
            stds[a] = float(np.sqrt(np.maximum(1e-9, ctx @ cov @ ctx)))
            evidences[a] = float(np.trace(self.A_mat[a]) - self.dim * self.lambda_reg)

        # softmax is only for the displayed probability; the decision uses the intervals
        exp_m = np.exp(np.clip(means * self.temp, -20.0, 20.0))
        p_dist = exp_m / np.sum(exp_m)

        order = np.argsort(means)[::-1]
        top, runner = int(order[0]), int(order[1])

        z_crit = norm.ppf(1.0 - (1.0 - self.ci) / 2.0)
        lo_top = float(means[top] - z_crit * stds[top])
        hi_top = float(means[top] + z_crit * stds[top])
        hi_run = float(means[runner] + z_crit * stds[runner])

        evidence = float(np.mean(evidences))
        separated = (lo_top > hi_run)
        enough = (evidence >= self.min_evidence)

        reported = top if (separated and enough) else None
        return reported, top, float(p_dist[top]), lo_top, hi_top, evidence

    def observe(self, input_vector, true_action, prev):
        """Online update towards the confirmed true action (reward 1)."""
        if self.feature_cols is not None and len(input_vector) == len(self.feature_cols):
            z = self._project(np.asarray(input_vector, dtype=float).reshape(1, -1))[0]
        else:
            z = np.asarray(input_vector[:self.n_pca], dtype=float)
        ctx = self._build_context(z, int(prev))
        a = int(true_action)
        self.A_mat[a] += np.outer(ctx, ctx)
        self.b_vec[a] += ctx
        return a

    def evaluate(self, df):
        preds, guesses, truths, probs, los, his, evs = [], [], [], [], [], [], []
        for _label, Z, truth in self._episodes(df):
            prev = START_TOKEN
            for t in range(len(truth)):
                rep, guess, p, lo, hi, ev = self.decide(Z[t], prev if self.memory else START_TOKEN)
                preds.append(rep)
                guesses.append(guess)
                truths.append(int(truth[t]))
                probs.append(p)
                los.append(lo)
                his.append(hi)
                evs.append(ev)
                prev = guess
        return dict(pred=np.array(preds, dtype=object), guess=np.array(guesses),
                    truth=np.array(truths), prob=np.array(probs),
                    lo=np.array(los), hi=np.array(his), evidence=np.array(evs))

    def effective_prev(self, prev):
        return prev if self.memory else START_TOKEN

    def responsibilities_for(self, feature_vector):
        """Project a raw feature vector to PCA space."""
        return self._project(np.asarray(feature_vector, dtype=float).reshape(1, -1))[0]

    def evidence_per_state(self):
        """Total evidence collected by each action."""
        return np.array([np.trace(A) for A in self.A_mat])


# older name, still used by train.py
HierarchicalBayesianRL = ContinuousBayesianRL


def calibration(probs, correct, n_bins=10):
    """Reliability bins + expected calibration error."""
    probs = np.asarray(probs, float)
    correct = np.asarray(correct, float)
    if len(probs) == 0:
        return [], float("nan")
    edges = np.linspace(0, 1, n_bins + 1)
    rows, ece = [], 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (probs > lo) & (probs <= hi) if i else (probs >= lo) & (probs <= hi)
        if not m.any():
            continue
        rows.append(dict(lo=lo, hi=hi, n=int(m.sum()),
                         confidence=float(probs[m].mean()),
                         accuracy=float(correct[m].mean())))
        ece += (m.sum() / len(probs)) * abs(correct[m].mean() - probs[m].mean())
    return rows, float(ece)
