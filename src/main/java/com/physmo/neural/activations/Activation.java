package com.physmo.neural.activations;

import com.physmo.neural.NodeLayer;

public interface Activation {
    double Activate(Double value);
    double Derivative(Double value);

    void LayerActivation(NodeLayer nl);

    void LayerDerivative(NodeLayer nl);
}