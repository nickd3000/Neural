package com.physmo.neural;

import com.physmo.neural.activations.ActivationType;

import java.util.ArrayList;
import java.util.List;

// Rewriting neural network class with what I've learned...

public class NN2 {

    private static final String NEWLINE = "\n";
    final List<NodeLayer> nodeLayers = new ArrayList<>();
    final List<WeightLayer> weightLayers = new ArrayList<>();
    private double learningRate = 0.1;
    private double dampenValue = 0.95;
    private double inputScale = 1;
    private double inputShift = 0;
    private double outputScale = 1;
    private double outputShift = 0;
    private double combinedError = 0;

    public NN2() {

    }

    public NN2 addLayer(int size, ActivationType actTp) {
        size += 1; // Add bias node.

        NodeLayer lastLayer = getLastNodeLayer();
        int newLayerId = nodeLayers.size() + 1;
        int requiredWeights = 0;

        if (lastLayer != null) {
            requiredWeights = lastLayer.size * size;
        }

        // RNN add an extra weight to each node
        //requiredWeights+=(size-1);

        NodeLayer newNodeLayer = new NodeLayer(size, newLayerId);
        newNodeLayer.activationType = actTp;

        nodeLayers.add(newNodeLayer);
        if (requiredWeights > 0) weightLayers.add(new WeightLayer(requiredWeights, lastLayer, newNodeLayer));

        return this;
    }

    private NodeLayer getLastNodeLayer() {
        int numNodeLayers = nodeLayers.size();
        if (numNodeLayers < 1) return null;
        return nodeLayers.get(numNodeLayers - 1);
    }

    public NN2 randomizeWeights(double low, double high) {
        double span = high - low;
        for (WeightLayer wl : weightLayers) {
            for (int i = 0; i < wl.weights.length; i++) {
                wl.weights[i] = low + (Math.random() * span);
            }
        }
        return this;
    }

    public NN2 learningRate(double value) {
        this.learningRate = value;
        return this;
    }

    public NN2 dampenValue(double value) {
        this.dampenValue = value;
        return this;
    }

    public NN2 activationType(ActivationType actTp) {
        NodeLayer nl = getLastNodeLayer();
        nl.activationType = actTp;
        return this;
    }


    public NN2 inputMapping(double inputScale, double inputShift) {
        this.inputScale = inputScale;
        this.inputShift = inputShift;
        return this;
    }

    public NN2 outputMapping(double outputScale, double outputShift) {
        this.outputScale = outputScale;
        this.outputShift = outputShift;
        return this;
    }

    public void setInputValue(int i, double v) {
        nodeLayers.get(0).values[i] = mapValue(v, inputScale, inputShift);
    }

    public void setOutputTargetValue(int i, double v) {
        getLastNodeLayer().targets[i] = mapValue(v, inputScale, inputShift);
    }

    public double getOutputValue(int i) {
        return unmapValue(getLastNodeLayer().values[i], outputScale, outputShift);
        //return getLastNodeLayer().derivatives[i];
    }
    public double getOutputValueRaw(int i) {
        return getLastNodeLayer().values[i];
        //return getLastNodeLayer().derivatives[i];
    }
    public double getInnerValue(int layer, int node) {
        NodeLayer nl = nodeLayers.get(layer);
        return nl.values[node];
    }

    public double getInnerDerivative(int layer, int node) {
        NodeLayer nl = nodeLayers.get(layer);
        return nl.derivatives[node];
    }

    @Override
    public String toString() {
        String output = "NN2\n";

        for (NodeLayer nl : nodeLayers) {
            output += "NodeLayer " + nl.layerId + " size " + nl.size + NEWLINE;
        }
        for (WeightLayer wl : weightLayers) {
            output += "WeightLayer size" + wl.size + NEWLINE;
        }

        return output;
    }

    public void run(boolean learn) {
        feedForward();

        backpropogate();

        if (learn) {
            learn();
        }
    }


    // STEP 1
    // Feed input values forward and calculate output values and errors.
    public void feedForward() {
        //activateLayer(nodeLayers.get(0));

        for (WeightLayer wl : weightLayers) {
            resetBiasNode(wl.sourceNodeLayer);
            propogateLayerPair(wl);
            activateLayer(wl.targetNodeLayer);
        }

        nodeLayers.forEach((nl) -> calculateLayerDerivatives(nl));

        calculateLayerErrors(getLastNodeLayer());
        combinedError = sumLayerError(getLastNodeLayer());
    }

    // STEP 2
    // Back propogate and calculate deltas.
    public void backpropogate() {
        for (int i = weightLayers.size() - 1; i >= 0; i--) {
            backPropogateLayerPair(weightLayers.get(i));
        }

        for (WeightLayer wl : weightLayers) {
            dampenDeltas(wl, dampenValue);
            updateDeltas(wl);
        }
    }

    // STEP 3
    public void learn() {
        weightLayers.forEach((wl) -> applyDeltasToWeights(wl));
    }


    public double getCombinedError() {
        return combinedError;
    }

    private void propogateLayerPair(WeightLayer wl) {

        wl.targetNodeLayer.clearValues();
        //wl.targetNodeLayer.addPreviousValues_experimental(); // instead of clearing, copy previous iterations value.

        double[] sourceValues = wl.sourceNodeLayer.values;
        double[] targetValues = wl.targetNodeLayer.values;
        double[] weights = wl.weights;

        int w = 0;

        for (double sourceValue : sourceValues) {
            for (int tv = 0; tv < targetValues.length; tv++) {
                targetValues[tv] += sourceValue * weights[w++];
            }
        }

        //wl.targetNodeLayer.storePreviousValues_experimental();
    }

    private void backPropogateLayerPair(WeightLayer wl) {

        double[] weights = wl.weights;
        double[] sourceErrors = wl.sourceNodeLayer.errors;
        double[] sourceValues = wl.sourceNodeLayer.values;
        double[] sourceDerivatives = wl.sourceNodeLayer.derivatives;
        double[] targetErrors = wl.targetNodeLayer.errors;

        int w = 0;
        wl.sourceNodeLayer.clearErrors();

        for (int sv = 0; sv < sourceErrors.length; sv++) {
            for (double targetError : targetErrors) {
                // NJD: commented out sourceDerivatives and binary classifier did not break...
                sourceErrors[sv] += /* sourceDerivatives[sv] * */ weights[w++] * targetError;
            }
        }

    }

    private void dampenDeltas(WeightLayer wl, double multiplier) {
        double[] deltas = wl.deltas;

        for (int d = 0; d < deltas.length; d++) {

            deltas[d] *= multiplier;

        }
    }

    // Add error to delta value.
    private void updateDeltas(WeightLayer wl) {
        double[] deltas = wl.deltas;
        double[] sourceErrors = wl.sourceNodeLayer.errors;
        double[] targetErrors = wl.targetNodeLayer.errors;
        double[] targetDerivatives = wl.targetNodeLayer.derivatives;
        double[] sourceValues = wl.sourceNodeLayer.values;

        int w = 0;

        for (int sv = 0; sv < sourceErrors.length; sv++) {
            for (int tv = 0; tv < targetErrors.length; tv++) {
                double delta = targetErrors[tv] * sourceValues[sv] * learningRate;
                delta *= targetDerivatives[tv];
                deltas[w] += delta;
                w++;
            }
        }
    }

    // Adjust weights using deltas.
    private void applyDeltasToWeights(WeightLayer wl) {
        double[] weights = wl.weights;
        double[] deltas = wl.deltas;

        int w = 0;

        for (int i = 0; i < weights.length; i++) {
            weights[i] += deltas[i];
        }
    }


    private void activateLayer(NodeLayer nl) {
        //nl.addPreviousValues_experimental();
        nl.activationType.getInstance().LayerActivation(nl);
        nl.storePreviousValues_experimental();
    }

    private void calculateLayerDerivatives(NodeLayer nl) {
        nl.activationType.getInstance().LayerDerivative(nl);
    }

    // derivatives are required before this is executed.
    private void calculateLayerErrors(NodeLayer nl) {
        for (int i = 0; i < nl.values.length; i++) {
            double e = nl.targets[i] - nl.values[i];
            nl.errors[i] = e;
        }
    }

    private double sumLayerError(NodeLayer nl) {
        double sum = 0;
        for (int i = 0; i < nl.values.length; i++) {
            sum += Math.abs(nl.errors[i]);
        }
        return sum;
    }

    private void resetBiasNode(NodeLayer nl) {
        int numNodes = nl.size;
        nl.values[numNodes - 1] = 1.0f;
    }

    private double mapValue(double val, double scale, double shift) {
        return (val * scale) + shift;
    }

    private double unmapValue(double val, double scale, double shift) {
        return (val / scale) + shift;
    }

    public void setWeightsFromArray(double [] weights) {
        int count = 0;
        for (WeightLayer wl : weightLayers) {
            for (int i=0;i<wl.size;i++) {
                wl.weights[i] = weights[count++];
            }
        }
    }

    public int getNumberOfWeights() {
        int count = 0;
        for (WeightLayer wl : weightLayers) {
            count += wl.size;
        }
        return count;
    }
}
