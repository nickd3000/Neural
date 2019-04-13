package com.physmo.neural.activations;

import com.physmo.neural.NodeLayer;

class Sigmoid implements Activation {

    @Override
    public void CalculateActivation(NodeLayer nl) {
        for (int i = 0; i < nl.size; i++) {
            nl.values[i] = 1.0 / (1 + Math.exp(-1.0 * nl.values[i]));
        }
    }

    @Override
    public void CalculateDerivative(NodeLayer nl) {
        for (int i = 0; i < nl.size; i++) {
            if (nl.values[i] < 0) nl.derivatives[i] = 0.1;
            else nl.derivatives[i] = nl.values[i] * (1.0 - nl.values[i]);
        }
    }
}
