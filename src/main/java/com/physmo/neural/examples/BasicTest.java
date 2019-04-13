package com.physmo.neural.examples;

import com.physmo.neural.NN2;
import com.physmo.neural.activations.ActivationType;

import java.text.DecimalFormat;

public class BasicTest {

    static DecimalFormat doubleFormat = new DecimalFormat("#.00");
    public static double[] inputs = {0.2, 0.4, 0.6,0.8};
    public static double[] outputs = {0.2, 0.9, 0.8,0.6};
    public static double[] calculated = {0,0,0,0};

    public static void main(String args[]) {

        ActivationType activationType = ActivationType.TANH;

        NN2 nn2 = new NN2()
                .addLayer(1, activationType)
                .addLayer(3, activationType)
                .addLayer(1, activationType)
                .randomizeWeights(-0.9, 0.9)
                .learningRate(0.001).dampenValue(0.9);

        double errorTotal = 0;

        for (int i = 0; i < 500000; i++) {

            errorTotal = 0;

            // Learning
            for (int k = 0; k < inputs.length; k++) {
                nn2.setInputValue(0, inputs[k]);
                nn2.setOutputTargetValue(0, outputs[k]);
                nn2.run(true);
                calculated[k] = nn2.getOutputValue(0);

                errorTotal += nn2.getCombinedError();
            }

            if (every(i, 1000)) System.out.println(displaySummary(nn2));

            if (errorTotal < 0.01) {
                //System.out.println(displaySummary(net));
                break;
            }
        }
    }

    public static String displaySummary(NN2 net) {
        String lst = "";
        for (double s : calculated) {lst+=String.format("%.3f ", s);}

        String str = String.format(
                "Error: %.3f output: %s",
                net.getCombinedError(),
                lst);
        //System.out.println("Error: " + net.errorTotal + "output:" + net.getOutput(0, 0, 1));
        return str;
    }

    public static boolean every(int x, int i) {
        if (x == 0) return true;
        if (x % i == 0) return true;
        return false;
    }

}
