package com.physmo.neural.activations;

import com.physmo.neural.NodeLayer;

class Linear implements Activation {

    @Override
    public double Activate(Double value) {
        return value;
    }

    @Override
    public double Derivative(Double value) {
        return 1;
    }

    @Override
    public void LayerActivation(NodeLayer nl) {
        for (int i = 0; i < nl.size; i++) {
            nl.values[i] = nl.values[i];
        }
    }

    @Override
    public void LayerDerivative(NodeLayer nl) {
        for (int i = 0; i < nl.size; i++) {
            nl.derivatives[i] = 1;
        }
    }
}
