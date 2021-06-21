package com.physmo.neural.activations;

import com.physmo.neural.NodeLayer;

// TODO: this activation requires more work, can't be done purely as single functions
class Softmax implements Activation {
    @Override
    public double Activate(Double value) {
        return 0;
    }

    @Override
    public double Derivative(Double value) {
        return 0;
    }
    @Override
    public void LayerActivation(NodeLayer nl) {

        double sum = 0;
        double max = -100;
        // add all outputs.
        for (int i = 0; i < nl.size; i++) {
            //double val = nl.values[i];
            nl.values[i] = nl.values[i] * nl.values[i];
            //val = Math.abs(val);
            //sum += Math.exp(val);
            //if (val>max) max=val;
            sum += nl.values[i];
        }
        if (sum < 0.001) sum = 0.001;
        // scale outputs by sum?
        for (int i = 0; i < nl.size; i++) {
            //double val = nl.values[i];
            //nl.values[i]=(Math.exp(val))/sum;
            nl.values[i] = nl.values[i] / sum;
        }
    }

    @Override
    public void LayerDerivative(NodeLayer nl) {
        for (int i = 0; i < nl.size; i++) {
            nl.derivatives[i] = nl.values[i] * (1.0 - nl.values[i]);
        }
    }
}
