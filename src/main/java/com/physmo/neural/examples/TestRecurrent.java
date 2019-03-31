package com.physmo.neural.examples;

import com.physmo.neural.NeuralNet;
import com.physmo.minvio.BasicDisplay;
import com.physmo.minvio.BasicDisplayAwt;
import com.physmo.minvio.BasicGraph;

import java.awt.*;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;

// add function to copy hidden layer to inputs (with an offset)
// functionality to pick best output (so there is only one)?
// 
public class TestRecurrent {
	private static final char GENERATION_SEED_CHAR = ' ';
	static int midLayerSize = 120 ; // 50
	static int charRange=95;
	static boolean skipCopy = false;
	static double scaleMin=-0.95;
	static double scaleMax=0.95;
	static int deltaCount = 1; // 1 number of learnings to combine for inertia
	static double randomAmount=0.010;
	BasicDisplay bd = null;
	BasicGraph scoreGraph = null;
	double learningError = 0;
	
	public void run() {
		bd = new BasicDisplayAwt(320, 240);
		scoreGraph = new BasicGraph(100);
		
		String book = loadTextFile("fox.txt");
		//String book = loadTextFile("sherlock.txt");
		//String book = loadTextFile("sphynx.txt");
		//String book = loadTextFile("abcd.txt");
		//String book = loadTextFile("wiki.txt");
		
		if (book.length()>0) System.out.println(book.substring(0, 200));
		
		NeuralNet net = new NeuralNet();
		String buildStr = ""+(charRange+midLayerSize+2)+" "+midLayerSize+" 100 100 "+charRange;
		net.buildNet(buildStr); //"286 60 256");
		net.learningRate=0.001;
		net.momentum=0.9945;
		net.randomiseAllWeights(-0.15, 0.15);
		net.softmaxOutput=false;

		long lastUpdate = System.currentTimeMillis();

		for (int m=0;m<80000;m++) {
			
			for (int i=0;i<50;i++) {
				learn(net, book, 50); // 2
			}
			
			if (System.currentTimeMillis()-lastUpdate>1000) {
				System.out.println("Iteration:"+m);
				lastUpdate = System.currentTimeMillis();
				generateOutput(net, 140);
				
				bd.cls(Color.white);
				scoreGraph.draw(bd, 10, 10, 300, 200, null);
				bd.refresh();
			}
		}
	}
	
	public void learn(NeuralNet net, String corpus, int batchSize) {
		
		char inChar=' ';
		char outChar=' ';
		int charPos=0;
		int uncommitted = 0;
		charPos = (int)(Math.random()*(double)(corpus.length()-batchSize));
		learningError=0;
		for (int i=0;i<batchSize;i++) {
			inChar=corpus.charAt(charPos);
			outChar=corpus.charAt(charPos+1);
			
			setInputFromChar(net, inChar);
			setExpectedOutputFromChar(net, outChar);
			copyInnerLayerToInput(net, midLayerSize, charRange);
			
			net.run();
			net.calculateLearningDeltas();
			learningError += net.errorTotal;
			
			uncommitted++;
			if (uncommitted>=deltaCount) {
				net.applyWeightDeltas(); 
				uncommitted=0;
			}
			
			charPos++;
		}
	}

	public void generateOutput(NeuralNet net, int size) {
		char prevChar = GENERATION_SEED_CHAR;
		System.out.println("Sample output:");
		
		for (int i=0;i<size;i++) {
			setInputFromChar(net, prevChar);
			copyInnerLayerToInput(net, midLayerSize, charRange);
			net.run();
			
			prevChar = getOutputChar(net);
			System.out.print(prevChar);
		}
		System.out.print("\n " + learningError + " ");
		
		if (scoreGraph!=null) {
			scoreGraph.addData(learningError);
		}
	}
	
	public void copyInnerLayerToInput(NeuralNet net, int numInnerNodes, int inputNodeOffset) {
		//mergeOutputToInput(net);
		
		if (skipCopy) return;
		double range = scaleMax;;
		for (int i=0;i<numInnerNodes;i++) {
			//double val = net.getInnerValue(1, i, -range, range);
			double val = net.getInnerValue(1, i);
			//val = Math.max(0, val);
			net.setInput(i+inputNodeOffset, val, scaleMin,scaleMax);
		}
	}
	
	public void mergeOutputToInput(NeuralNet net) {
		int outputLayer = 3;
		int numOutputNodes = charRange;
		double range = scaleMax;;
		for (int i=0;i<numOutputNodes;i++) {
			double val = net.getInnerValue(outputLayer, i, -range, range);
			val = Math.max(0, val);
			double inp = net.getInnerValue(0, i, -range, range);
			net.setInput(i+0, val+inp, scaleMin,scaleMax);
		}
	}
	
	public char getOutputChar(NeuralNet net) {
		return getOutputCharWeighted(net);
		/*
		double maxVal=-10;
		int maxId=0;
		for (int i=0;i<numChars;i++) {
			double val = net.getOutput(i, scaleMin, scaleMax);
			if (val>maxVal) {
				maxVal = val;
				maxId=i;
			}
		}
		return (char)mapIntToChar(maxId);
		*/
	}
	
	// version with roulette style weighted selection.
	public char getOutputCharWeighted(NeuralNet net) {

		int maxId=0;
		double total=0;
		double val=0;
		double softMax=0;
		
		for (int i=0;i<charRange;i++) {
			//val=Math.max(0,net.getOutput(i, scaleMin, scaleMax));
			val=Math.max(0,net.getOutput(i));
			softMax+=Math.exp(Math.abs(val));
			val=val*val*val;
			total += val;
		}
		
		double pick = Math.random()*total;
			
		double runningTotal=0,previousRunningTotal=0;
		for (int i=0;i<charRange;i++) {
			//val=Math.max(0,net.getOutput(i, scaleMin, scaleMax));
			val=Math.max(0,net.getOutput(i));
			val=val*val*val;
			
			runningTotal += val;
			//runningTotal += (Math.exp(Math.abs(val)))/softMax;
			
			if (val>0 && pick>=previousRunningTotal &&
				pick<=runningTotal) {
				maxId=i;
				break;
			}

			previousRunningTotal = runningTotal;
		}
		return (char)mapIntToChar(maxId);
	}

	
	public void setInputFromChar(NeuralNet net, char c) {
		int ic = mapCharToInt(c);
		for (int i=0;i<charRange;i++) {
			if (i==ic)  {
				net.setInput(i, scaleMax, scaleMin, scaleMax);
			}
			else {
				net.setInput(i, 0, scaleMin, scaleMax);
			}
		}
	}
	public void setExpectedOutputFromChar(NeuralNet net, char c) {
		int ic = mapCharToInt(c);
		for (int i=0;i<charRange;i++) {
			if (i==ic)  {
				net.setTarget(i, scaleMax, scaleMin, scaleMax);
			}
			else {
				net.setTarget(i, 0, scaleMin, scaleMax);
			}
		}
	}
	
	// 32-127=95
	public void testMapCharToInt(char c) {
		int i = mapCharToInt(c);
		
		System.out.println("char: "+c+" int:"+i);
	}
	public int mapCharToInt(char c) {
		int i = (int)c;
		i=i-32;
		if (i<0) i=0;
		if (i>=charRange) i=charRange;
		
		return i;
	}
	public char mapIntToChar(int i) {
		if (i<0) i=0;
		if (i>charRange) i=charRange;
		i+=32;
		
		return (char)(i);
	}
	
	public String loadTextFile(String fileName) {
		String content = "";
		try {
			content = new String(Files.readAllBytes(Paths.get(fileName)));
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return content;
	}
}
