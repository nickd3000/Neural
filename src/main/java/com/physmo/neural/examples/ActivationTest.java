package com.physmo.neural.examples;

import com.physmo.minvio.BasicDisplay;
import com.physmo.minvio.BasicDisplayAwt;
import com.physmo.neural.NodeLayer;
import com.physmo.neural.activations.ActivationType;

import java.awt.*;

public class ActivationTest {
    Color colBackground = new Color(7, 23, 29);
    Color colBorder = new Color(120, 120, 120);
    Color colAxes = new Color(100, 100, 100);
    Color colAct = new Color(200, 100, 100);
    Color colDer = new Color(100, 100, 200);

    public static void main(String[] args) {
        ActivationTest app = new ActivationTest();
        app.run();
    }

    public void run() {
        BasicDisplay bd = new BasicDisplayAwt(640, 400);
        bd.setTitle("Activation functions");
        bd.cls(colBackground);

        renderActivationFunction(bd, ActivationType.LINEAR, 10, 10, 300, 100);
        renderActivationFunction(bd, ActivationType.RELU, 10, 30 + 100, 300, 100);
        renderActivationFunction(bd, ActivationType.SIGMOID, 10, 50 + 200, 300, 100);

        renderActivationFunction(bd, ActivationType.SOFTMAX, 320, 10, 300, 100);
        renderActivationFunction(bd, ActivationType.TANH, 320, 30 + 100, 300, 100);
        renderActivationFunction(bd, ActivationType.NONE, 320, 50 + 200, 300, 100);
    }

    public void renderActivationFunction(BasicDisplay bd, ActivationType a, int x, int y, int w, int h) {
        bd.setDrawColor(colBorder);
        bd.drawRect(x, y, x + w, y + h);
        bd.setDrawColor(colAxes);
        bd.drawLine(x, y + h / 2, x + w, y + h / 2);
        bd.drawLine(x + w / 2, y, x + w / 2, y + h);

        NodeLayer dummyLayer = new NodeLayer(1, 1);


        double v = 0;
        double act = 0;
        double der = 0;
        double widthScaler = w;
        double heightScaler = h / 3;

        for (int v1 = 0; v1 < w; v1++) {
            v = (((double) v1 - (w / 2)) / (double) w) * 8.0;
            dummyLayer.values[0] = v;
            a.getInstance().CalculateActivation(dummyLayer);
            a.getInstance().CalculateDerivative(dummyLayer);
            act = clamp(dummyLayer.values[0], -1.5, 1.5);
            der = clamp(dummyLayer.derivatives[0], -1.5, 1.5);
            bd.setDrawColor(colAct);
            bd.drawFilledRect(
                    (int) (x + (v1)),
                    (int) (y + (h / 2) - (act * heightScaler)), 2, 2);
            bd.setDrawColor(colDer);
            bd.drawFilledRect(
                    (int) (x + (v1)),
                    (int) (y + (h / 2) - (der * heightScaler)), 2, 2);
        }

        bd.refresh();
    }

    public double clamp(double val, double min, double max) {
        if (val < min) return min;
        if (val > max) return max;
        return val;
    }
}
