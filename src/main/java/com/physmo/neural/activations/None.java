package com.physmo.neural.activations;

import com.physmo.neural.NodeLayer;

class None implements Activation {
    @Override
    public double Activate(Double value) {
        return 0;
    }

    @Override
    public double Derivative(Double value) {
        return 1;
    }

    @Override
    public void LayerActivation(NodeLayer nl) {
    }

    @Override
    public void LayerDerivative(NodeLayer nl) {
        for (int i = 0; i < nl.size; i++) {
            nl.derivatives[i] = 1;//nl.values[i];
        }
    }
}
