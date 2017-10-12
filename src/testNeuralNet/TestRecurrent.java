package testNeuralNet;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;

import Neural.NeuralNet;

// add function to copy hidden layer to inputs (with an offset)
// functionality to pick best output (so there is only one)?
// 
public class TestRecurrent {
	static int midLayerSize = 80; // 50
	static int numChars=95;
	static boolean skipCopy = false;
	static double scaleMin=-0.99;
	static double scaleMax=0.99;
	static int deltaCount = 1; // number of learnings to combine for inertia
	static double randomAmount=0.010;
	
	public void run() {
		//String book = loadTextFile("fox.txt");
		String book = loadTextFile("sherlock.txt");
		//String book = loadTextFile("sphynx.txt");
		//String book = loadTextFile("abcd.txt");
		//String book = loadTextFile("wiki.txt");
		
		if (book.length()>0) System.out.println(book.substring(0, 200));
		
		NeuralNet net = new NeuralNet();
		String buildStr = ""+(numChars+midLayerSize+2)+" "+midLayerSize+" 80 80 "+numChars;
		net.buildNet(buildStr); //"286 60 256");
		net.learningRate=0.0013*10.1;
		net.momentum=0.45;
		net.randomiseAllWeights(-0.5, 0.5);
		net.softmaxOutput=false;
		
		for (int m=0;m<10000;m++) {
			System.out.println("Iteration:"+m);
			for (int i=0;i<200;i++) {
				learn(net, book, 2);
			}
			generateOutput(net, 140);
		}
	}
	
	public void generateOutput(NeuralNet net, int size) {
		char prevChar = ' ';
		System.out.println("Sample output:");
		for (int i=0;i<size;i++) {
			setInputFromChar(net, prevChar);
			copyInnerLayerToInput(net, midLayerSize, numChars);
			net.run();
			prevChar = getOutputChar(net);
			System.out.print(prevChar);
		}
		System.out.print("\n");
	}
	
	public void copyInnerLayerToInput(NeuralNet net, int numInnerNodes, int inputNodeOffset) {
		if (skipCopy) return;
		double range = scaleMax; //2;
		for (int i=0;i<numInnerNodes;i++) {
			double val = net.getInnerValue(1, i, -range, range);
			net.setInput(i+inputNodeOffset, val, scaleMin,scaleMax);
		}
	}
	
	public void learn(NeuralNet net, String corpus, int batchSize) {
		
		char inChar=' ';
		char outChar=' ';
		int charPos=0;
		int uncommitted = 0;
		charPos = (int)(Math.random()*(double)(corpus.length()-batchSize));
		for (int i=0;i<batchSize;i++) {
			inChar=corpus.charAt(charPos);
			outChar=corpus.charAt(charPos+1);
			
			setInputFromChar(net, inChar);
			setOutputFromChar(net, outChar);
			copyInnerLayerToInput(net, midLayerSize, numChars);
			
			net.run();
			net.learn();
			uncommitted++;
			if (uncommitted>=deltaCount) {
				net.applyWeightDeltas(); 
				uncommitted=0;
			}
			
			charPos++;
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
		
		for (int i=0;i<numChars;i++) {
			val=Math.max(0,net.getOutput(i, scaleMin, scaleMax));
			val=val*val*val;
			total += val;
		}
		
		double pick = Math.random()*total;
			
		double runningTotal=0,previousRunningTotal=0;
		for (int i=0;i<numChars;i++) {
			val=Math.max(0,net.getOutput(i, scaleMin, scaleMax));
			val=val*val*val;
			runningTotal += val;
			
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
		for (int i=0;i<numChars;i++) {
			if (i==ic)  {
				net.setInput(i, scaleMax, scaleMin, scaleMax);
			}
			else {
				net.setInput(i, 0, scaleMin, scaleMax);
				//net.setInput(i, scaleMin+(Math.random()*randomAmount), scaleMin, scaleMax);
			}
		}
	}
	public void setOutputFromChar(NeuralNet net, char c) {
		int ic = mapCharToInt(c);
		for (int i=0;i<numChars;i++) {
			if (i==ic)  {
				net.setTarget(i, scaleMax, scaleMin, scaleMax);
			}
			else {
				net.setTarget(i, 0, scaleMin, scaleMax);
				//net.setTarget(i, scaleMin+(Math.random()*randomAmount), scaleMin, scaleMax);
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
		if (i>=numChars) i=numChars;
		
		return i;
	}
	public char mapIntToChar(int i) {
		if (i<0) i=0;
		if (i>numChars) i=numChars;
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
