package com.physmo.neural.activations;

import com.physmo.neural.NodeLayer;

class Relu implements Activation {
    @Override
    public double Activate(Double value) {
        return Math.max(0, value);
    }

    @Override
    public double Derivative(Double value) {
        if (value <= 0) return 0.1;
        return 1;
    }
    @Override
    public void LayerActivation(NodeLayer nl) {
        for (int i = 0; i < nl.size; i++) {
            nl.values[i] = Math.max(0, nl.values[i]);
        }
    }

    @Override
    public void LayerDerivative(NodeLayer nl) {
        for (int i = 0; i < nl.size; i++) {
            if (nl.values[i] <= 0) nl.derivatives[i] = 0.1;
            else nl.derivatives[i] = 1;
        }
    }
}