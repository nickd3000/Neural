package testNeuralNet;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;

import Neural.NeuralNet;

// add function to copy hidden layer to inputs (with an offset)
// functionality to pick best output (so there is only one)?
// 
public class TestRecurrent {

	public void run() {
		String book = loadTextFile("sherlock.txt");
		
		if (book.length()>0) System.out.println(book.substring(0, 2000));
		
		NeuralNet net = new NeuralNet();
		net.buildNet("286 30 30 256");
		net.learningRate=0.0013;
		net.momentum=0.45;
		net.randomiseAllWeights(-0.5, 0.5);
		
		for (int m=0;m<1000;m++) {
			System.out.println("Iteration:"+m);
			for (int i=0;i<100;i++) {
				learn(net, book, 1000);
			}
			generateOutput(net, 200);
		}
	}
	
	public void generateOutput(NeuralNet net, int size) {
		char prevChar = 'a';
		System.out.println("Sample output:");
		for (int i=0;i<size;i++) {
			setInputFromChar(net, prevChar);
			copyInnerLayerToInput(net, 30, 0xff);
			net.run();
			prevChar = getOutputChar(net);
			System.out.print(prevChar);
		}
		System.out.print("\n");
	}
	
	public void copyInnerLayerToInput(NeuralNet net, int numInnerNodes, int inputNodeOffset) {
		for (int i=0;i<numInnerNodes;i++) {
			double val = net.getInnerValue(1, i, -2.0, 2.0);
			net.setInput(i+inputNodeOffset, val, -2.0, 2.0);
		}
	}
	
	public void learn(NeuralNet net, String corpus, int batchSize) {
		
		char inChar=' ';
		char outChar=' ';
		int charPos=1000;
		int uncommitted = 0;
		charPos = (int)(Math.random()*(double)(corpus.length()-batchSize));
		for (int i=0;i<batchSize;i++) {
			inChar=corpus.charAt(charPos);
			outChar=corpus.charAt(charPos+1);
			
			setInputFromChar(net, inChar);
			setOutputFromChar(net, inChar);
			copyInnerLayerToInput(net, 30, 0xff);
			
			net.run();
			net.learn();
			uncommitted++;
			if (uncommitted>5) {
				net.applyWeightDeltas(); 
				uncommitted=0;
			}
			
			charPos++;
		}
	}
	
	public char getOutputChar(NeuralNet net) {
		double maxVal=-10;
		int maxId=0;
		for (int i=0;i<0xff;i++) {
			double val = net.getOutput(i, 0.0, 1.0);
			if (val>maxVal) {
				maxVal = val;
				maxId=i;
			}
		}
		return (char)maxId;
	}
	
	public void setInputFromChar(NeuralNet net, char c) {
		for (int i=0;i<0xff;i++) {
			if (i==(int)c)  {
				net.setInput(i, 1.0, 0.0, 1.0);
			}
			else {
				net.setInput(i, 0.1, 0.0, 1.0);
			}
		}
	}
	public void setOutputFromChar(NeuralNet net, char c) {
		for (int i=0;i<0xff;i++) {
			if (i==(int)c)  {
				net.setTarget(i, 1.0, 0.0, 1.0);
			}
			else {
				net.setTarget(i, 0.1, 0.0, 1.0);
			}
		}
	}
	
	public void testMapCharToInt(char c) {
		int i = mapCharToInt(c);
		System.out.println("char: "+c+" int:"+i);
	}
	public int mapCharToInt(char c) {
		int i = (int)c;
		return i;
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
