package com.physmo.reference;

import com.physmo.minvio.BasicDisplay;
import com.physmo.minvio.BasicDisplayAwt;
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
    static int midLayerSize = 55; //200 ; // 50
    static double learningRate = 0.001; // 0.01
    static double dampenValue = 0.000; // 0.6
    double learningError = 0;
    boolean dynamicLearningRate = false;
    BasicDisplay bd = null;
    BasicGraph scoreGraph = null;

    public static void main(String[] args) {
        TestRecurrentNN2 test = new TestRecurrentNN2();
        test.run();
    }

    public void run() {
        bd = new BasicDisplayAwt(320, 240);
        scoreGraph = new BasicGraph(100);

        //String book = loadTextFile("fox.txt");
        String book = loadTextFile("sherlock.txt");
        //String book = loadTextFile("sphynx.txt");
        //String book = loadTextFile("abcd.txt");
        //String book = loadTextFile("wiki.txt");

        if (book.length() > 0) System.out.println(book.substring(0, 200));


        NN2 net = new NN2()
                .addLayer(charRange + midLayerSize + 2, ActivationType.TANH)
//                .addLayer(midLayerSize, ActivationType.TANH)
//                .addLayer(midLayerSize/3, ActivationType.TANH)
//                .addLayer(midLayerSize/3, ActivationType.TANH)
//                .addLayer(midLayerSize, ActivationType.RELU)
//                .addLayer(midLayerSize, ActivationType.RELU)
//                .addLayer(midLayerSize, ActivationType.TANH)
                .addLayer(midLayerSize, ActivationType.RELU)
                .addLayer(charRange, ActivationType.LINEAR)
                .randomizeWeights(-0.001, 0.01)
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

            for (int i = 0; i < 15; i++) {
                learn(net, book, 10); //30 250); // 2
            }


            if (dynamicLearningRate && m % 100 == 0) {
                learningRate *= 0.97;
                net.learningRate(learningRate);
            }

            if (System.currentTimeMillis() - lastUpdate > 1000) {
                System.out.println("Iteration:" + m);
                lastUpdate = System.currentTimeMillis();
                generateOutput(net, 140);

                bd.cls(Color.white);
                scoreGraph.draw(bd, 10, 10, 300, 200, null);
                bd.drawText("LR=" + learningRate, 20, 200);
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

        for (int i = 0; i < batchSize; i++) {
            inChar = corpus.charAt(charPos);
            outChar = corpus.charAt(charPos + 1);

            setInputFromChar(net, inChar);
            setExpectedOutputFromChar(net, outChar);

            copyInnerLayerToInput(net, midLayerSize, charRange, i == 0);

            net.feedForward();

            net.backpropogate();

            learningError += net.getCombinedError();

            charPos++;
            net.learn();
        }


    }

    public void generateOutput(NN2 net, int size) {
        char prevChar = GENERATION_SEED_CHAR;
        prevChar = (char) ('a' + (char) (Math.random() * 20));

        System.out.println("Sample output:");

        for (int i = 0; i < size; i++) {
            System.out.print(prevChar);

            setInputFromChar(net, prevChar);

            copyInnerLayerToInput(net, midLayerSize, charRange, i == 0 ? true : false);
            //copyOutputLayerToInput(net, midLayerSize, charRange, i == 0 ? true : false);
            //net.run(false);
            net.feedForward();

            prevChar = getOutputCharWeighted(net);

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
        if (i > charRange) i = charRange;
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

        int maxId = 0;
        double total = 0;
        double val = 0;
        double softMax = 0;

        // Calculate total of all outputs.
        for (int i = 0; i < charRange; i++) {
            val = Math.abs(net.getOutputValue(i));
            total += transformWeightedValue(val);;
        }

        double pick = Math.random() * total;

        double runningTotal = 0, previousRunningTotal = 0;
        for (int i = 0; i < charRange; i++) {
            val = Math.abs(net.getOutputValue(i));

            runningTotal += transformWeightedValue(val);;

            if (val > 0 && pick > previousRunningTotal &&
                    pick <= runningTotal) {
                maxId = i;
                break;
            }

            previousRunningTotal = runningTotal;
        }
        return (char) mapIntToChar(maxId);
    }

    public void setInputFromChar(NN2 net, char c) {
        int ic = mapCharToInt(c);
        for (int i = 0; i < charRange; i++) {
            if (i == ic) {
                net.setInputValue(i, scaleMax); //, scaleMin, scaleMax);
            } else {
                net.setInputValue(i, 0.1);//, scaleMin, scaleMax);
            }
        }
    }

    public void setExpectedOutputFromChar(NN2 net, char c) {
        int ic = mapCharToInt(c);
        for (int i = 0; i < charRange; i++) {
            if (i == ic) {
                net.setOutputTargetValue(i, scaleMax);
            } else {
                net.setOutputTargetValue(i, 0);
            }
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
        if (i >= charRange) i = charRange;

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
