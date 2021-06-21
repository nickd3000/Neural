package com.physmo.neural.examples;

import com.physmo.minvio.BasicDisplay;
import com.physmo.minvio.BasicDisplayAwt;
import com.physmo.minvio.BasicGraph;
import com.physmo.neural.NN2;
import com.physmo.neural.NN2Renderer;
import com.physmo.neural.activations.ActivationType;

import java.awt.*;

// Test learning a curve.
public class FormulaTest {


    public static final double learningRate = 0.01;
    public static final double dampenValue = 0.1; //0.0059;

    public static void main(String[] args) {
        testFormula();
    }


    public static void testFormula() {
        BasicDisplay display = new BasicDisplayAwt(640, 480);
        BasicGraph graphError = new BasicGraph(2000);


        ActivationType att = ActivationType.TANH;
        NN2 net = new NN2()
                .addLayer(1, att)
                .addLayer(5, att)
                .addLayer(1, att)
                .randomizeWeights(-0.2, 0.2)
                .inputMapping(1, 0)
                .outputMapping(1, 0)
                .learningRate(learningRate)
                .dampenValue(dampenValue);

        NN2Renderer nn2renderer = new NN2Renderer(net, display, 330, 20, 300, 200);

        display.startTimer();

        for (int i = 0; i < 50000000; i++) {
            for (int e = 0; e < 10; e++) {

                double functionInput = Math.random() * 6.00;
                double functionResult = function(functionInput);

                net.setInputValue(0, functionInput);
                net.setOutputTargetValue(0, functionResult);
                net.feedForward();
                net.backpropogate();
            }

            net.learn();

            if (display.getEllapsedTime() > 1000 / 30) {
                display.startTimer();

                net.run(false);
                display.cls(new Color(15, 41, 69));
                int y = 0;
                double error = 0;
                for (int x = 0; x < 300; x++) {
                    net.setInputValue(0, (double) x / 50.0);
                    net.setOutputTargetValue(0, function((double) x / 50.0));
                    //net.run(false);
                    net.feedForward();

                    error += net.getCombinedError();
                    y = transformGraphValue(0);
                    display.setDrawColor(Color.gray);
                    display.drawRect(x, y,  2,  2);
                    y = transformGraphValue(function((double) x / 50.0));
                    display.setDrawColor(new Color(48, 155, 156));
                    display.drawRect(x, y,  2,  2);

                    y = transformGraphValue(net.getOutputValue(0));
                    //display.setDrawColor(new Color(193, 190, 89));
                    //display.drawRect(x, y, x + 2, y + 2);
                    drawGraphPoint(display,x,y,new Color(193, 190, 89));

//                    for (int k=0;k<5;k++) {
//                        y = transformGraphValue(net.getInnerValue(1, k));
//                        drawGraphPoint(display, x, y, display.getDistinctColor(k, 0.5));
//                    }
                }

                // Draw network.
                nn2renderer.draw();

                graphError.addData(error / 300.0);
                graphError.draw(display, 20, 170, 300, 300, Color.gray);

                display.repaint();
            }
        }
    }

    public static void drawGraphPoint(BasicDisplay bd, int x, int y, Color c) {
        bd.setDrawColor(c);
        bd.drawRect(x, y,  1,  1);
    }


    public static double function(double x) {
        //return (x-1)/5;
        //return Math.tanh(x);
        double val = (Math.sin(x * 1) * 0.4 + (Math.sin(x * 2) * 0.4)) + (Math.sin(x * 5) * 0.2);
        //return (Math.cos(x)*0.5)+0.5;
        //return (Math.cos(x*5)*0.5)+0.5;
        //if (val<0) val=-0.5;
        //if (val>0) val=0.5;
        return val;
    }

    public static int transformGraphValue(double val) {
        val = (val + 1.0) * 50.0;
        int ret = (int) val;
        if (ret < 0) ret = 5;
        if (ret > 200) ret = 195;
        ret += 25;
        return ret;
    }

    public static boolean every(int x, int i) {
        if (x == 0) return true;
        if (x % i == 0) return true;
        return false;
    }
}

