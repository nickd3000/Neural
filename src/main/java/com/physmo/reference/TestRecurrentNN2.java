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

// Simple character-level recurrent demo using manual feedback from a hidden layer.
public class TestRecurrentNN2 {

    // Configuration
    private static final int CHAR_RANGE = 95;          // ASCII 32..126 inclusive
    private static final int MID_LAYER_SIZE = 128;
    private static final double INITIAL_LEARNING_RATE = 0.0015;
    private static final double DAMPEN_VALUE = 0.95;
    private static final boolean DYNAMIC_LEARNING_RATE = true;
    private static final double LR_DECAY = 0.9997;

    // Input/target scaling for one-hot
    private static final double ONE_HOT_ON = 1.0;
    private static final double ONE_HOT_OFF = 0.0;

    // Recurrent feedback
    private static final boolean ENABLE_FEEDBACK_COPY = true;
    private static final int FEEDBACK_SOURCE_LAYER_INDEX = 2; // which hidden layer to feed back
    private static final double FEEDBACK_SCALE = 0.15;        // keep small to avoid runaway
    private static final String PRIMER = " the ";            // seed text for generation

    // UI / run-time
    private static final int OUTPUT_SAMPLE_LEN = 140;
    private static final long UI_UPDATE_MS = 1000L;

    private double learningError = 0;
    private double learningRate = INITIAL_LEARNING_RATE;

    private BasicDisplay bd = null;
    private DrawingContext dc = null;
    private BasicGraph scoreGraph = null;

    public static void main(String[] args) {
        new TestRecurrentNN2().run();
    }

    public void run() {
        bd = new BasicDisplayAwt(320, 240);
        dc = bd.getDrawingContext();
        scoreGraph = new BasicGraph(100);

        String book = loadTextFile("sherlock.txt");
        //String book = loadTextFile("fox.txt");
        //String book = loadTextFile("sphynx.txt");
        //String book = loadTextFile("abcd.txt");
        //String book = loadTextFile("wiki.txt");

        if (book.length() > 0) System.out.println(book.substring(0, 200));

        NN2 net = new NN2()
                .addLayer(CHAR_RANGE + MID_LAYER_SIZE + 2, ActivationType.TANH)
                .addLayer(MID_LAYER_SIZE, ActivationType.TANH)
                .addLayer(MID_LAYER_SIZE, ActivationType.TANH)
                .addLayer(MID_LAYER_SIZE, ActivationType.TANH)
                .addLayer(MID_LAYER_SIZE, ActivationType.TANH)
                .addLayer(MID_LAYER_SIZE, ActivationType.TANH)
                .addLayer(MID_LAYER_SIZE, ActivationType.TANH)
                .addLayer(CHAR_RANGE, ActivationType.SOFTMAX)
                .xavierWeights()
                .learningRate(learningRate)
                .dampenValue(DAMPEN_VALUE);

        long lastUpdate = System.currentTimeMillis();

        for (int m = 0; m < 80_000; m++) {
            for (int i = 0; i < 100; i++) {
                learn(net, book, 50);
            }

            if (DYNAMIC_LEARNING_RATE && m % 100 == 0) {
                learningRate *= LR_DECAY;
                net.learningRate(learningRate);
            }

            if (System.currentTimeMillis() - lastUpdate > UI_UPDATE_MS) {
                System.out.println("Iteration:" + m);
                lastUpdate = System.currentTimeMillis();
                generateOutput(net, OUTPUT_SAMPLE_LEN);

                dc.cls(Color.white);
                scoreGraph.draw(bd, 10, 10, 300, 200, null);
                dc.drawText("LR=" + learningRate, 20, 200);
                bd.repaint();
            }
        }
    }

    // Train for one random window in the corpus
    private void learn(NN2 net, String corpus, int batchSize) {
        int charPos = (int) (Math.random() * (double) (corpus.length() - batchSize));
        learningError = 0;

        net.clearIntermediateValues();

        for (int i = 0; i < batchSize; i++) {
            char inChar = corpus.charAt(charPos);
            char outChar = corpus.charAt(charPos + 1);

            setInputFromChar(net, inChar);
            setOneHotTarget(net, outChar);

            copyHiddenToRecurrentInputs(net, MID_LAYER_SIZE, CHAR_RANGE, i == 0);

            net.feedForward();
            net.backpropogate();

            learningError += net.getCombinedError();
            charPos++;

            // Mini-batch update
            if (i % 10 == 0 || i == batchSize - 1) {
                net.learn();
            }
        }
    }

    // Generate a sample string using primer text and softmax sampling
    private void generateOutput(NN2 net, int size) {
        System.out.println("Sample output:");

        // Warm up hidden state using primer (not printed)
        net.clearIntermediateValues();
        for (int i = 0; i < PRIMER.length(); i++) {
            char c = PRIMER.charAt(i);
            setInputFromChar(net, c);
            copyHiddenToRecurrentInputs(net, MID_LAYER_SIZE, CHAR_RANGE, i == 0);
            net.feedForward();
        }

        char prevChar = PRIMER.charAt(PRIMER.length() - 1);

        for (int i = 0; i < size; i++) {
            System.out.print(prevChar);

            setInputFromChar(net, prevChar);
            copyHiddenToRecurrentInputs(net, MID_LAYER_SIZE, CHAR_RANGE, false);
            net.feedForward();

            // Sample directly from softmax probabilities with temperature
            prevChar = sampleOutputWithTemperature(net, 0.9);
        }
        System.out.print("\n " + learningError + " ");

        if (scoreGraph != null) {
            scoreGraph.addData(learningError);
        }
    }

    // Copy a hidden layer activations into the recurrent inputs with scaling and optional clear
    private void copyHiddenToRecurrentInputs(NN2 net, int numHiddenNodes, int inputOffset, boolean clear) {
        if (!ENABLE_FEEDBACK_COPY) return;

        for (int i = 0; i < numHiddenNodes; i++) {
            double val = clear ? 0.0 : net.getInnerValue(FEEDBACK_SOURCE_LAYER_INDEX, i);
            val *= FEEDBACK_SCALE;
            val = Math.tanh(val); // bound values
            net.setInputValue(i + inputOffset, val);
        }
    }

    // One-hot encode the input character to the first CHAR_RANGE inputs
    private void setInputFromChar(NN2 net, char c) {
        int idx = mapCharToInt(c);
        for (int i = 0; i < CHAR_RANGE; i++) {
            net.setInputValue(i, i == idx ? ONE_HOT_ON : ONE_HOT_OFF);
        }
    }

    // One-hot targets for softmax output
    private void setOneHotTarget(NN2 net, char c) {
        int idx = mapCharToInt(c);
        for (int i = 0; i < CHAR_RANGE; i++) {
            net.setOutputTargetValue(i, i == idx ? ONE_HOT_ON : ONE_HOT_OFF);
        }
    }

    // Sample from the softmax outputs directly
    private char sampleOutput(NN2 net) {
        double[] probs = readSoftmaxOutputs(net);
        double pick = Math.random();
        double cumulative = 0.0;
        for (int i = 0; i < CHAR_RANGE; i++) {
            cumulative += probs[i];
            if (pick <= cumulative) return (char) mapIntToChar(i);
        }
        return (char) mapIntToChar(CHAR_RANGE - 1);
    }

    // Temperature applied to probabilities: p_i^(1/T) and renormalize
    private char sampleOutputWithTemperature(NN2 net, double temperature) {
        if (temperature <= 0.0 || Double.isNaN(temperature)) {
            return getOutputCharMax(net);
        }

        double invT = 1.0 / temperature;
        double[] probs = readSoftmaxOutputs(net);
        double sum = 0.0;
        for (int i = 0; i < CHAR_RANGE; i++) {
            probs[i] = Math.pow(Math.max(0.0, probs[i]), invT);
            sum += probs[i];
        }
        if (sum <= 0.0 || Double.isNaN(sum) || Double.isInfinite(sum)) {
            return getOutputCharMax(net);
        }
        for (int i = 0; i < CHAR_RANGE; i++) probs[i] /= sum;

        double pick = Math.random();
        double cumulative = 0.0;
        for (int i = 0; i < CHAR_RANGE; i++) {
            cumulative += probs[i];
            if (pick <= cumulative) return (char) mapIntToChar(i);
        }
        return (char) mapIntToChar(CHAR_RANGE - 1);
    }

    // Read outputs as probabilities (assumes softmax final layer)
    private double[] readSoftmaxOutputs(NN2 net) {
        double[] probs = new double[CHAR_RANGE];
        double sum = 0.0;
        for (int i = 0; i < CHAR_RANGE; i++) {
            double p = net.getOutputValue(i);
            if (p < 0.0) p = 0.0; // soft clamp in case of numeric drift
            probs[i] = p;
            sum += p;
        }
        if (sum > 0) {
            for (int i = 0; i < CHAR_RANGE; i++) probs[i] /= sum;
        } else {
            // fallback to uniform
            double u = 1.0 / CHAR_RANGE;
            for (int i = 0; i < CHAR_RANGE; i++) probs[i] = u;
        }
        return probs;
    }

    private char getOutputCharMax(NN2 net) {
        double maxVal = Double.NEGATIVE_INFINITY;
        int maxId = 0;
        for (int i = 0; i < CHAR_RANGE; i++) {
            double val = net.getOutputValue(i);
            if (val > maxVal) {
                maxVal = val;
                maxId = i;
            }
        }
        return (char) mapIntToChar(maxId);
    }

    // ASCII mapping helpers: map 32..126 -> 0..94 and back
    private static int mapCharToInt(char c) {
        int i = ((int) c) - 32;
        if (i < 0) i = 0;
        if (i >= CHAR_RANGE) i = CHAR_RANGE - 1;
        return i;
    }

    private static int mapIntToChar(int i) {
        if (i < 0) i = 0;
        if (i >= CHAR_RANGE) i = CHAR_RANGE - 1;
        return i + 32;
    }

    private String loadTextFile(String fileName) {
        String content = "";
        try {
            final URL resource = TestRecurrentNN2.class.getClassLoader().getResource(fileName);
            Path path = Paths.get(resource.getPath());
            content = new String(Files.readAllBytes(path));
        } catch (IOException e) {
            e.printStackTrace();
        }
        return content;
    }
}
