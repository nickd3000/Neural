package com.physmo.neural.activations;

import com.physmo.neural.NodeLayer;

// Softmax over non-bias units with stable exponentiation.
// Derivative set to 1.0 to let (target - prob) flow as in softmax+cross-entropy.
class Softmax implements Activation {
    @Override
    public double Activate(Double value) {
        // Not used for vector activations.
        return value;
    }

    @Override
    public double Derivative(Double value) {
        // Not used for vector activations.
        return 1.0;
    }

    @Override
    public void LayerActivation(NodeLayer nl) {
        int n = nl.size;
        if (n <= 1) return;

        int last = n - 1; // treat as bias slot
        // Find max logit (exclude bias)
        double maxLogit = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < last; i++) {
            double v = nl.values[i];
            if (v > maxLogit) maxLogit = v;
        }

        // Exponentiate shifted logits and sum (exclude bias)
        double sum = 0.0;
        for (int i = 0; i < last; i++) {
            double e = Math.exp(nl.values[i] - maxLogit);
            nl.values[i] = e;
            sum += e;
        }
        if (sum == 0.0 || Double.isNaN(sum) || Double.isInfinite(sum)) {
            // Degenerate case: fall back to uniform distribution
            double uniform = 1.0 / Math.max(1, last);
            for (int i = 0; i < last; i++) {
                nl.values[i] = uniform;
            }
        } else {
            // Normalize
            for (int i = 0; i < last; i++) {
                nl.values[i] /= sum;
            }
        }

        // Leave bias value as-is (do not normalize it)
        // If you want a deterministic bias on outputs, consider forcing it to 1.0 here.
        // nl.values[last] = 1.0;
    }

    @Override
    public void LayerDerivative(NodeLayer nl) {
        int n = nl.size;
        if (n <= 1) return;

        int last = n - 1; // bias
        // In this framework, deltas are computed as:
        //   delta = targetError * sourceValue * learningRate * targetDerivative
        // For softmax + cross-entropy, the effective derivative of the activation cancels,
        // so we set 1.0 to pass (target - prob) unchanged.
        for (int i = 0; i < last; i++) {
            nl.derivatives[i] = 1.0;
        }
        // No gradient flows through the bias activation here
        nl.derivatives[last] = 0.0;
    }
}
