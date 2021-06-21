package com.physmo.neural.activations;

import com.physmo.neural.NodeLayer;

class Tanh implements Activation {
    @Override
    public double Activate(Double value) {
        return Math.tanh(value);
    }

    @Override
    public double Derivative(Double value) {
        return (1 - (value * value));
    }

    @Override
    public void LayerActivation(NodeLayer nl) {
        for (int i = 0; i < nl.size; i++) {
            nl.values[i] = Math.tanh(nl.values[i]);
        }
    }

    @Override
    public void LayerDerivative(NodeLayer nl) {
        for (int i = 0; i < nl.size; i++) {
            nl.derivatives[i] = (1 - (nl.values[i] * nl.values[i]));
        }
    }
}
