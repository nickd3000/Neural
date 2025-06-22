package com.physmo.neural.activations;

import com.physmo.neural.NodeLayer;

class Relu implements Activation {
    double slope = 0.1;
    double max = 1;

    @Override
    public double Activate(Double value) {
        if (value <= 0) return value * slope;
        if (value >= 5) return value * slope;
        return value;
    }

    @Override
    public double Derivative(Double value) {
        if (value <= 0 || value > max) return slope;
        return 1;
    }

    @Override
    public void LayerActivation(NodeLayer nl) {
        for (int i = 0; i < nl.size; i++) {
            if (nl.values[i] <= 0) nl.values[i] = nl.values[i] * slope;
            if (nl.values[i] > max) nl.values[i] = nl.values[i] * slope;
            //nl.values[i] = Math.max(nl.values[i]*slope, nl.values[i]);
        }
    }

    @Override
    public void LayerDerivative(NodeLayer nl) {
        for (int i = 0; i < nl.size; i++) {
            if (nl.values[i] <= 0 || nl.values[i] > max) nl.derivatives[i] = slope;
            else nl.derivatives[i] = 1;
        }
    }
}