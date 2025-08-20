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
    // Momentum coefficient: velocity v = beta * v + grad; weights += lr * v
    private double momentum = 0.9;
    // Optional absolute gradient clip applied before accumulating into momentum (null = disabled)
    private Double gradClipAbs = null;

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

    public NN2 xavierWeights() {
        for (WeightLayer wl : weightLayers) {
            int fanIn = wl.sourceNodeLayer.size;
            int fanOut = wl.targetNodeLayer.size;
            double limit = Math.sqrt(2.0 / fanIn); // He initialization for ReLU

            for (int i = 0; i < wl.weights.length; i++) {
                wl.weights[i] = (Math.random() * 2 * limit) - limit;
            }
        }
        return this;
    }


    public NN2 learningRate(double value) {
        this.learningRate = value;
        return this;
    }

    /**
     * Sets the dampening (momentum) coefficient used to decay the per‑weight velocity.
     * Interpreted as the classical momentum parameter beta in [0, 1):
     * - Higher values (e.g., 0.9–0.99) retain more past velocity, yielding smoother and potentially faster convergence,
     *   but can overshoot and may require a smaller learning rate.
     * - Lower values (e.g., 0.0–0.5) forget history faster, making updates more responsive but noisier.
     * Use 0.0 for no momentum. This method is kept for backward compatibility and also forwards to momentum(value).
     *
     * @param value momentum coefficient (beta), typically around 0.9
     * @return this instance for chaining
     */
    public NN2 dampenValue(double value) {
        this.dampenValue = value;
        // Keep backwards compatibility: map to momentum semantics
        this.momentum = value;
        return this;
    }

    public NN2 momentum(double beta) {
        this.momentum = beta;
        return this;
    }

    public NN2 gradientClipAbs(double clipAbs) {
        this.gradClipAbs = clipAbs;
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
        // Use output mapping for targets, not input mapping
        getLastNodeLayer().targets[i] = mapValue(v, outputScale, outputShift);
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
        if (!learn) {
            // Inference-only: do not touch deltas or weights
            feedForward();
            return;
        }
        feedForward();
        backpropogate();
        learn();
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

        nodeLayers.forEach(this::calculateLayerDerivatives);

        calculateLayerErrors(getLastNodeLayer());
        combinedError = sumLayerError(getLastNodeLayer());
    }

    // STEP 2
    // Back propagate and calculate deltas.
    public void backpropogate() {
        // Convert output errors into deltas in-place: delta = error * f'(z)
        NodeLayer out = getLastNodeLayer();
        int outBiasIdx = out.size - 1;
        for (int i = 0; i < outBiasIdx; i++) {
            out.errors[i] *= out.derivatives[i];
        }
        // Ensure bias error/delta is zero
        if (out.errors.length > 0) out.errors[outBiasIdx] = 0.0;

        // Backpropagate deltas through all weight layers
        for (int i = weightLayers.size() - 1; i >= 0; i--) {
            backPropogateLayerPair(weightLayers.get(i));
        }

        // Classical momentum on per-weight "velocity" stored in wl.deltas:
        // v = beta * v + grad
        for (WeightLayer wl : weightLayers) {
            dampenDeltas(wl, momentum);
            updateDeltas(wl);
        }
    }

    // STEP 3
    public void learn() {
        weightLayers.forEach(this::applyDeltasToWeights);
    }

    public void clearIntermediateValues() {
        for (NodeLayer nl : nodeLayers) {
            nl.clearValues();
        }
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

                if (isBad(targetValues[tv])) {
                    System.out.println("Bad target value: " + targetValues[tv]);
                }

            }
        }

        //wl.targetNodeLayer.storePreviousValues_experimental();
    }

    private void backPropogateLayerPair(WeightLayer wl) {

        double[] weights = wl.weights;
        double[] sourceErrors = wl.sourceNodeLayer.errors;          // will become source deltas after multiplying by source derivatives
        double[] sourceDerivatives = wl.sourceNodeLayer.derivatives;
        double[] targetDeltas = wl.targetNodeLayer.errors;          // already multiplied by target derivatives
        int sourceSize = wl.sourceNodeLayer.size;
        int targetSize = wl.targetNodeLayer.size;

        // Exclude bias neuron of target from propagation
        int targetActive = targetSize - 1;
        int sourceActive = sourceSize - 1;

        int w = 0;

        wl.sourceNodeLayer.clearErrors();

        // sourceErrors = W * targetDeltas (excluding target bias)
        for (int sv = 0; sv < sourceSize; sv++) {
            double acc = 0.0;
            for (int tv = 0; tv < targetActive; tv++) {
                acc += weights[w++] * targetDeltas[tv];
            }
            // Skip the bias weight column into target bias node (we maintain a dense matrix; advance w over it)
            w += 1; // advance over weight into target bias for this source neuron
            sourceErrors[sv] += acc;
        }

        // Convert source errors into source deltas for the next iteration
        for (int sv = 0; sv < sourceActive; sv++) {
            sourceErrors[sv] *= sourceDerivatives[sv];
        }
        // Bias delta is zero
        sourceErrors[sourceActive] = 0.0;
    }

    private void dampenDeltas(WeightLayer wl, double multiplier) {
        double[] deltas = wl.deltas;

        for (int d = 0; d < deltas.length; d++) {
            deltas[d] *= multiplier;
        }
    }

    // Accumulate raw gradient into velocity (stored in wl.deltas).
    // After dampenDeltas() we add current grad: v = beta*v + grad.
    private void updateDeltas(WeightLayer wl) {
        double[] deltas = wl.deltas;                    // serves as velocity v
        double[] sourceValues = wl.sourceNodeLayer.values;
        double[] targetDeltas = wl.targetNodeLayer.errors; // already includes activation derivative
        int sourceSize = wl.sourceNodeLayer.size;
        int targetSize = wl.targetNodeLayer.size;
        int targetActive = targetSize - 1; // exclude target bias from weight updates

        int w = 0;

        for (int sv = 0; sv < sourceSize; sv++) {
            for (int tv = 0; tv < targetActive; tv++) {
                double grad = targetDeltas[tv] * sourceValues[sv]; // raw gradient dL/dw

                // Optional per-element gradient clipping before adding to momentum
                if (gradClipAbs != null) {
                    double cap = gradClipAbs.doubleValue();
                    if (grad > cap) grad = cap;
                    else if (grad < -cap) grad = -cap;
                }

                deltas[w] += grad; // v = beta*v + grad
                w++;
            }
            // Skip the weight into target bias (do not update it)
            w++;
        }
    }

    public boolean isBad(double d) {
        return Double.isNaN(d) || Double.isInfinite(d);
    }


    // Adjust weights using velocity and learning rate: w += lr * v
    private void applyDeltasToWeights(WeightLayer wl) {
        double[] weights = wl.weights;
        double[] deltas = wl.deltas; // velocity v

        for (int i = 0; i < weights.length; i++) {
            weights[i] += learningRate * deltas[i];
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
        int biasIdx = nl.size - 1;
        for (int i = 0; i < biasIdx; i++) {

            if (isBad(nl.targets[i])) {
                System.out.println("Bad target: " + nl.targets[i]);
            }
            if (isBad(nl.values[i])) {
                System.out.println("Bad values: " + nl.values[i]);
            }

            double e = nl.targets[i] - nl.values[i];

            if (e > 100) {
                e = 100;
                System.out.println("Error too large: " + e);
            }

            nl.errors[i] = e;

            if (isBad(e)) {
                System.out.println("Bad error: " + e);
            }
        }
        // No error/target for bias neuron
        nl.errors[biasIdx] = 0.0;
    }

    private double sumLayerError(NodeLayer nl) {
        double sum = 0;
        int biasIdx = nl.size - 1;
        for (int i = 0; i < biasIdx; i++) {
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
        // Correct inverse of mapValue(val) = val * scale + shift
        if (scale == 0.0) return val; // avoid division by zero; alternatively throw
        return (val - shift) / scale;
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

    // Resets momentum/velocity (useful between runs or when changing lr/momentum)
    public void clearDeltas() {
        for (WeightLayer wl : weightLayers) {
            for (int i = 0; i < wl.deltas.length; i++) {
                wl.deltas[i] = 0.0;
            }
        }
    }
}
