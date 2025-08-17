package com.physmo.reference;

import com.physmo.minvio.BasicDisplay;
import com.physmo.minvio.BasicDisplayAwt;
import com.physmo.minvio.DrawingContext;
import com.physmo.neural.NodeLayer;
import com.physmo.neural.activations.ActivationType;

import java.awt.*;

public class ActivationVisualizer {
    Color colBackground = new Color(7, 23, 29);
    Color colBorder = new Color(120, 120, 120);
    Color colAxes = new Color(100, 100, 100);
    Color colAct = new Color(200, 100, 100);
    Color colDer = new Color(100, 100, 200);

    public static void main(String[] args) {
        ActivationVisualizer app = new ActivationVisualizer();
        app.run();
    }

    public void run() {
        BasicDisplay bd = new BasicDisplayAwt(800, 600);
        DrawingContext dc = bd.getDrawingContext();

        bd.setTitle("Activation functions");
        bd.getDrawingContext().cls(colBackground);

        int padding = 10;
        int padding2 = padding*2;
        int panelWidth = (bd.getWidth()/2) - (padding * 2);
        int panelHeight = (bd.getHeight()/3) - (padding * 2);

        renderActivationFunction(dc, ActivationType.LINEAR, padding, padding, panelWidth, panelHeight);
        renderActivationFunction(dc, ActivationType.RELU, padding, (padding2+panelHeight), panelWidth, panelHeight);
        renderActivationFunction(dc, ActivationType.SIGMOID, padding, (padding2+panelHeight)*2, panelWidth, panelHeight);

        renderActivationFunction(dc, ActivationType.SOFTMAX, padding2+panelWidth, padding, panelWidth, panelHeight);
        renderActivationFunction(dc, ActivationType.TANH, padding2+panelWidth, (padding2+panelHeight), panelWidth, panelHeight);
        renderActivationFunction(dc, ActivationType.NONE, padding2+panelWidth, (padding2+panelHeight)*2, panelWidth, panelHeight);

        bd.repaint();
    }

    public void renderActivationFunction(DrawingContext dc, ActivationType a, int x, int y, int w, int h) {
        dc.setDrawColor(colBorder);
        dc.drawRect(x, y,  w,  h);
        dc.setDrawColor(colAxes);
        dc.drawLine(x, y + h / 2, x + w, y + h / 2);
        dc.drawLine(x + w / 2, y, x + w / 2, y + h);

        NodeLayer dummyLayer = new NodeLayer(1, 1);


        double v = 0;
        double act = 0;
        double der = 0;
        double widthScaler = w;
        double heightScaler = h / 3.0;

        for (int v1 = 0; v1 < w; v1++) {
            v = (((double) v1 - (w / 2.0)) / (double) w) * 8.0;
            dummyLayer.values[0] = v;
            a.getInstance().LayerActivation(dummyLayer);
            a.getInstance().LayerDerivative(dummyLayer);
            act = clamp(dummyLayer.values[0], -1.5, 1.5);
            der = clamp(dummyLayer.derivatives[0], -1.5, 1.5);
            dc.setDrawColor(colAct);
            dc.drawFilledRect(
                    (int) (x + (v1)),
                    (int) (y + (h / 2.0) - (act * heightScaler)), 2, 2);
            dc.setDrawColor(colDer);
            dc.drawFilledRect(
                    (int) (x + (v1)),
                    (int) (y + (h / 2.0) - (der * heightScaler)), 2, 2);
        }


        dc.drawText(a.name(), x + 10, y + 20);
    }

    public double clamp(double val, double min, double max) {
        if (val < min) return min;
        if (val > max) return max;
        return val;
    }
}
