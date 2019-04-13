package com.physmo.neural;

import com.physmo.neural.activations.ActivationType;

public class NodeLayer {
    public int size;
    public int layerId;
    public ActivationType activationType;

    public double[] values;
    public double[] derivatives;
    public double[] targets;
    public double[] errors;
    public double[] previousValues; // experimental

    public NodeLayer(int size, int layerId) {
        this.size = size;
        this.layerId = layerId;
        activationType = ActivationType.TANH;
        values = new double[size];
        derivatives = new double[size];
        targets = new double[size];
        errors = new double[size];
        previousValues = new double[size];
    }

    public void clearValues() {
        for (int i = 0; i < values.length; i++) {
            values[i] = 0;
            derivatives[i] = 0;
            errors[i] = 0;
            previousValues[i] = 0;
        }
    }

    public void clearErrors() {
        for (int i = 0; i < errors.length; i++) {
            errors[i] = 0;
        }
    }

    // see what happens if we just add the previous value back in
    public void addPreviousValues_experimental() {
        for (double value : values) {
            //values[i]+=previousValues[i]/50;

        }
    }

    public void storePreviousValues_experimental() {
        for (int i = 0; i < values.length; i++) {
            previousValues[i] = values[i];
        }
    }
}