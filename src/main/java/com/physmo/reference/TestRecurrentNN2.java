package com.physmo.reference;

import com.physmo.minvio.BasicDisplay;
import com.physmo.minvio.BasicDisplayAwt;
import com.physmo.minvio.DrawingContext;
import com.physmo.minvio.utils.BasicGraph;
import com.physmo.neural.NN2;
import com.physmo.neural.activations.ActivationType;

import java.awt.Color;
import java.io.IOException;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

// add function to copy hidden layer to inputs (with an offset)
// functionality to pick best output (so there is only one)?
// 
public class TestRecurrentNN2 {
    private static final char GENERATION_SEED_CHAR = ' ';

    static int charRange = 95;
    static boolean skipCopy = false;
    static double scaleMin = -0.95;
    static double scaleMax = 0.95;
    static int deltaCount = 1; // 1 number of learnings to combine for inertia
    static double randomAmount = 0.010;
    static int midLayerSize = 64; //200 ; // 50
    static double learningRate = 0.015; // 0.01
    static double dampenValue = 0.9;//.9; // 0.6
    double learningError = 0;
    boolean dynamicLearningRate = true;
    BasicDisplay bd = null;
    DrawingContext dc = null;
    BasicGraph scoreGraph = null;

    public static void main(String[] args) {
        TestRecurrentNN2 test = new TestRecurrentNN2();
        test.run();
    }

    public void run() {
        bd = new BasicDisplayAwt(320, 240);
        dc = bd.getDrawingContext();
        scoreGraph = new BasicGraph(100);

        //String book = loadTextFile("fox.txt");
        //String book = loadTextFile("sherlock.txt");
        //String book = loadTextFile("sphynx.txt");
        //String book = loadTextFile("abcd.txt");
        String book = loadTextFile("wiki.txt");

        if (book.length() > 0) System.out.println(book.substring(0, 200));


        NN2 net = new NN2()
                .addLayer(charRange + midLayerSize + 2, ActivationType.TANH)
                .addLayer(midLayerSize, ActivationType.TANH)
                .addLayer(midLayerSize, ActivationType.TANH)
                .addLayer(midLayerSize, ActivationType.TANH)
                .addLayer(midLayerSize, ActivationType.TANH)
                .addLayer(midLayerSize, ActivationType.TANH)
                .addLayer(midLayerSize, ActivationType.TANH)
                .addLayer(charRange, ActivationType.SOFTMAX)
                .xavierWeights()
                .learningRate(learningRate)
                .dampenValue(dampenValue);

//		String buildStr = ""+(charRange+midLayerSize+2)+" "+midLayerSize+" 100 100 "+charRange;
//		net.buildNet(buildStr); //"286 60 256");
//		net.learningRate=0.001;
//		net.momentum=0.9945;
//		net.randomiseAllWeights(-0.15, 0.15);
//		net.softmaxOutput=false;

        long lastUpdate = System.currentTimeMillis();

        for (int m = 0; m < 80000; m++) {

            for (int i = 0; i < 50; i++) {
                learn(net, book, 20); //30 250); // 2
            }


            if (dynamicLearningRate && m % 100 == 0) {
                learningRate *= 0.99;
                net.learningRate(learningRate);
            }

            if (System.currentTimeMillis() - lastUpdate > 1000) {
                System.out.println("Iteration:" + m);
                lastUpdate = System.currentTimeMillis();
                generateOutput(net, 140);

                dc.cls(Color.white);
                scoreGraph.draw(bd, 10, 10, 300, 200, null);
                dc.drawText("LR=" + learningRate, 20, 200);
                bd.repaint();
            }
        }
    }

    public void learn(NN2 net, String corpus, int batchSize) {

        char inChar = ' ';
        char outChar = ' ';
        int charPos = 0;
        int uncommitted = 0;
        charPos = (int) (Math.random() * (double) (corpus.length() - batchSize));
        learningError = 0;

        //net.clearIntermediateValues();

        for (int i = 0; i < batchSize; i++) {
            inChar = corpus.charAt(charPos);
            outChar = corpus.charAt(charPos + 1);

            setInputFromChar(net, inChar);
            setExpectedOutputFromChar(net, outChar);

            copyInnerLayerToInput2(net, midLayerSize, charRange, i == 0);

            net.feedForward();

            net.backpropogate();

            learningError += net.getCombinedError();

            charPos++;

            // Only learn every N steps, not every character
            if (i % 10 == 0 || i == batchSize - 1) {
                net.learn();
            }

        }


    }

    public void generateOutput(NN2 net, int size) {
        // Prime with a short, common sequence to build state before sampling
        String primer = " the ";
        char prevChar = primer.charAt(primer.length() - 1);

        System.out.println("Sample output:");

        // Feed primer without printing it (except last char becomes the starting prevChar)
        for (int i = 0; i < primer.length(); i++) {
            char c = primer.charAt(i);
            setInputFromChar(net, c);
            copyInnerLayerToInput2(net, midLayerSize, charRange, i == 0);
            net.feedForward();
        }

        // Now generate and print
        for (int i = 0; i < size; i++) {
            System.out.print(prevChar);

            setInputFromChar(net, prevChar);
            copyInnerLayerToInput2(net, midLayerSize, charRange, false);
            net.feedForward();

            // Sample from the current softmax distribution (with temperature)
            prevChar = getOutputCharWithTemperature(net, 0.5);
            // Alternatively, greedy:
            // prevChar = getOutputCharMax(net);
        }
        System.out.print("\n " + learningError + " ");

        if (scoreGraph != null) {
            scoreGraph.addData(learningError);
        }
    }

    public void copyInnerLayerToInput(NN2 net, int numInnerNodes, int inputNodeOffset, boolean clear) {
        //mergeOutputToInput(net);

        if (skipCopy) return;
        double range = scaleMax;
        int innerLayerIndex = 2;
        for (int i = 0; i < numInnerNodes; i++) {
            //double val = net.getInnerValue(1, i, -range, range);
            double val = net.getInnerValue(innerLayerIndex, i);
            //val = Math.max(0, val);
            if (clear) val = 0;

            net.setInputValue(i + inputNodeOffset, val); //, scaleMin,scaleMax);
        }
    }

    // with scaling to prevent runaway feedback
    public void copyInnerLayerToInput2(NN2 net, int numInnerNodes, int inputNodeOffset, boolean clear) {
        if (skipCopy) return;

        int innerLayerIndex = 2;
        for (int i = 0; i < numInnerNodes; i++) {
            double val = net.getInnerValue(innerLayerIndex, i);
            if (clear) val = 0;

            // Scale the recurrent connection
            val *= 0.15; // or even smaller like 0.1
            val = Math.tanh(val); // Bound the values

            net.setInputValue(i + inputNodeOffset, val);
        }
    }


    public void copyOutputLayerToInput(NN2 net, int numInnerNodes, int inputNodeOffset, boolean clear) {
        //mergeOutputToInput(net);

        if (skipCopy) return;
        double range = scaleMax;
        for (int i = 0; i < charRange; i++) {
            //double val = net.getInnerValue(1, i, -range, range);
            double val = net.getOutputValue(i);
            //val = Math.max(0, val);
            if (clear) val = 0;

            net.setInputValue(i + inputNodeOffset, val); //, scaleMin,scaleMax);
        }
    }

    public char getOutputChar(NN2 net) {
        //return getOutputCharWeighted(net);

        double maxVal = -10;
        int maxId = 0;
        for (int i = 0; i < charRange; i++) {
            double val = Math.abs(net.getOutputValue(i));
            //double val = net.getOutputValue(i);
            //val += Math.random() * 0.1;
            if (val > maxVal) {
                maxVal = val;
                maxId = i;
            }
        }
        return (char) mapIntToChar(maxId);

    }

    public char mapIntToChar(int i) {
        if (i < 0) i = 0;
        if (i >= charRange) i = charRange - 1;
        i += 32;

        return (char) (i);
    }

    public double transformWeightedValue(double val) {
        val = (1+val) * (1+val);
        if (val<1.125) val = 0;
        return val;
    }

    // version with roulette style weighted selection.
    public char getOutputCharWeighted(NN2 net) {
        // Softmax outputs already represent a probability distribution (sum ≈ 1)
        // Sample directly from them rather than using abs/transform.
        double[] probs = new double[charRange];
        double sum = 0.0;
        for (int i = 0; i < charRange; i++) {
            double p = net.getOutputValue(i);
            if (p < 0) p = 0; // safety clamp
            probs[i] = p;
            sum += p;
        }
        if (sum <= 0.0 || Double.isNaN(sum)) {
            return getOutputCharMax(net);
        }
        // Normalize in case of drift
        for (int i = 0; i < charRange; i++) probs[i] /= sum;

        double pick = Math.random();
        double cumulative = 0.0;
        for (int i = 0; i < charRange; i++) {
            cumulative += probs[i];
            if (pick <= cumulative) {
                return (char) mapIntToChar(i);
            }
        }
        return (char) mapIntToChar(charRange - 1);
    }

    public char getOutputCharWithTemperature(NN2 net, double temperature) {
        // Apply temperature to probabilities: p_i^(1/T) / sum_j p_j^(1/T)
        if (temperature <= 0.0 || Double.isNaN(temperature)) {
            return getOutputCharMax(net);
        }

        double invT = 1.0 / temperature;
        double[] scaled = new double[charRange];
        double sum = 0.0;

        for (int i = 0; i < charRange; i++) {
            double p = net.getOutputValue(i); // softmax probability
            if (p < 0) p = 0;                 // clamp negatives
            double v = Math.pow(p, invT);     // temperature scaling on probs
            if (Double.isNaN(v) || Double.isInfinite(v)) v = 0.0;
            scaled[i] = v;
            sum += v;
        }

        if (sum <= 0.0 || Double.isNaN(sum) || Double.isInfinite(sum)) {
            return getOutputCharMax(net);
        }

        for (int i = 0; i < charRange; i++) {
            scaled[i] /= sum;
        }

        double pick = Math.random();
        double cumulative = 0.0;
        for (int i = 0; i < charRange; i++) {
            cumulative += scaled[i];
            if (pick <= cumulative) {
                return (char) mapIntToChar(i);
            }
        }
        return (char) mapIntToChar(charRange - 1);
    }

    public char getOutputCharMax(NN2 net) {
        double maxVal = Double.NEGATIVE_INFINITY;
        int maxId = 0;
        for (int i = 0; i < charRange; i++) {
            double val = net.getOutputValue(i);
            if (val > maxVal) {
                maxVal = val;
                maxId = i;
            }
        }
        return (char) mapIntToChar(maxId);
    }


    public void setInputFromChar(NN2 net, char c) {
        int ic = mapCharToInt(c);
        for (int i = 0; i < charRange; i++) {
            if (i == ic) {
                net.setInputValue(i, scaleMax); //, scaleMin, scaleMax);
            } else {
                net.setInputValue(i, 0.0);//, scaleMin, scaleMax);
            }
        }
    }

    public void setExpectedOutputFromChar(NN2 net, char c) {
        // One-hot targets for softmax: 1.0 at the true class, 0.0 elsewhere
        int ic = mapCharToInt(c);
        for (int i = 0; i < charRange; i++) {
            net.setOutputTargetValue(i, i == ic ? 1.0 : 0.0);
        }
    }

    // 32-127=95
    public void testMapCharToInt(char c) {
        int i = mapCharToInt(c);

        System.out.println("char: " + c + " int:" + i);
    }

    public int mapCharToInt(char c) {
        int i = (int) c;
        i = i - 32;
        if (i < 0) i = 0;
        if (i >= charRange) i = charRange - 1;

        return i;
    }

    public String loadTextFile(String fileName) {
        String content = "";
        try {
            final URL resource = TestRecurrentNN2.class.getClassLoader().getResource(fileName);
            Path path = Paths.get(resource.getPath());
            content = new String(Files.readAllBytes(path));
            //content = new String(Files.readAllBytes(Paths.get(fileName)));
        } catch (IOException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
        return content;
    }
}
