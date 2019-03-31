package com.physmo.neural.examples;

import com.physmo.neural.NN2;
import com.physmo.neural.activations.ActivationType;

import java.text.DecimalFormat;

public class BasicTest {

    static DecimalFormat doubleFormat = new DecimalFormat("#.00");
    public static double[] inputs = {0.5, 0.2, 0.4};
    public static double[] outputs = {0.15, 0.72, 0.24};

    public static void main(String args[]) {
        NN2 nn2 = new NN2()
                .addLayer(1,ActivationType.SIGMOID)
                .activationType(ActivationType.SIGMOID)
                .addLayer(3,ActivationType.SIGMOID)
                .activationType(ActivationType.SIGMOID)
                .addLayer(1,ActivationType.SIGMOID)
                .activationType(ActivationType.SIGMOID)
                .randomizeWeights(-0.9, 0.9)
                .learningRate(0.01).dampenValue(0.9);

        double errorTotal = 0;

        for (int i = 0; i < 500000; i++) {

            errorTotal = 0;

            // Learning
            for (int k = 0; k < inputs.length; k++) {
                nn2.setInputValue(0, inputs[k]);
                nn2.setOutputTargetValue(0, outputs[k]);
                nn2.run(true);
                errorTotal += nn2.getCombinedError();
            }

            if (every(i, 1000)) System.out.println(displaySummary(nn2));

            if (errorTotal < 0.001) {
                //System.out.println(displaySummary(net));
                break;
            }
        }
    }

    public static String displaySummary(NN2 net) {
        String str = String.format(
                "Error: %.3f output: %f.3",
                net.getCombinedError(),
                net.getOutputValue(0));
        //System.out.println("Error: " + net.errorTotal + "output:" + net.getOutput(0, 0, 1));
        return str;
    }

    public static boolean every(int x, int i) {
        if (x == 0) return true;
        if (x % i == 0) return true;
        return false;
    }

}
