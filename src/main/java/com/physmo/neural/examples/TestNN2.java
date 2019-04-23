package com.physmo.neural.examples;

import com.physmo.minvio.BasicDisplay;
import com.physmo.minvio.BasicDisplayAwt;
import com.physmo.minvio.BasicGraph;
import com.physmo.neural.NN2;
import com.physmo.neural.activations.ActivationType;

import java.awt.*;
import java.text.DecimalFormat;

public class TestNN2 {
    public static final int numIterations = 2000;
    static DecimalFormat doubleFormat = new DecimalFormat("#.00");
    BasicDisplay bd = new BasicDisplayAwt(400, 400);
    BasicGraph graph = new BasicGraph(2000);

    public static void main(String[] args) {
        TestNN2 app = new TestNN2();
        app.basicTest();
    }

    public void basicTest() {
        NN2 nn2 = new NN2()
                .addLayer(2, ActivationType.RELU)
                .activationType(ActivationType.RELU)
                .addLayer(2, ActivationType.RELU)
                .activationType(ActivationType.RELU)
                .addLayer(2, ActivationType.RELU)
                .activationType(ActivationType.RELU)
                .randomizeWeights(-0.9, 0.9)
                .learningRate(0.01).dampenValue(0.5);

        nn2.setInputValue(0, 1);
        nn2.setInputValue(1, 0);
        nn2.setOutputTargetValue(0, 0);
        nn2.setOutputTargetValue(1, 1);

        System.out.println(nn2.toString());

        for (int i = 0; i < numIterations; i++) {
            if (Math.random() > 0.5) {
                nn2.setInputValue(0, 1);
                nn2.setInputValue(1, 0);
                nn2.setOutputTargetValue(0, 0);
                nn2.setOutputTargetValue(1, 1);
            } else {
                nn2.setInputValue(0, 0);
                nn2.setInputValue(1, 1);
                nn2.setOutputTargetValue(0, 1);
                nn2.setOutputTargetValue(1, 0);
            }

            nn2.run(true);
            System.out.println("error: " + doubleFormat.format(nn2.getCombinedError()));

            graph.addData(nn2.getCombinedError());

            bd.cls(Color.white);
            graph.draw(bd, 1, 1, 400, 400, null);

            bd.refresh();
        }


    }
}
