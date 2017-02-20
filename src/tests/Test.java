package tests;

import Neural.NeuralNet;
import ToolBox.BasicDisplay;
import ToolBox.BasicGraph;

import java.awt.*;

public class Test {

	public static void main(String[] args) {
		//testBasic();
		testSin();
		//testMapping();
		
		//TestBinaryClassifier testBC = new TestBinaryClassifier();
		//testBC.run();
		
		//TestAudio testAudio = new TestAudio();
		//testAudio.run();
	}

	public static void testMapping() {
		NeuralNet net = new NeuralNet();
		System.out.println("mapValue(-1,-1,1): " + net.mapValue(-1, -1, 1));
		System.out.println("mapValue(1,-1,1): " + net.mapValue(1, -1, 1));
		System.out.println("unmapValue(0.1,-1,1): " + net.unmapValue(0.1, -1, 1));
		System.out.println("unmapValue(0.9,-1,1): " + net.unmapValue(0.9, -1, 1));
	}
	
	public static void testBasic() {
		NeuralNet net = new NeuralNet();
		net.buildNet("1 30 1");
		for (int i=0;i<5000;i++) {
			net.setInput(0, 0, 0, 1);
			//net.setInput(1, -1);
			net.setTarget(0, 0.1234, 0, 1);
			//net.setTarget(1, 0.0);
			net.run();
			net.learn();
			//System.out.println("Out1:" + net.getOutput(0) + " Out2:" + net.getOutput(1));
			net.run();
			if (every(i,1000)) System.out.println("Error: " + net.errorTotal + "output:" + net.getOutput(0, 0, 1));
		}
	}
	
	public static void testSin() {
		BasicDisplay display = new BasicDisplay(640, 480);
		BasicGraph graphError = new BasicGraph(20000);
		NeuralNet net = new NeuralNet();
		net.buildNet("1 8 1");
		net.randomiseAllWeights(-2, 2);
		net.learningRate = 0.0015;
		net.momentum = 0.65;
		
		for (int i=0;i<50000000;i++) {
			for (int e=0;e<10;e++) {
				double rnd = Math.random()*9.00;
				double res = function(rnd);
				net.setInput(0, rnd, 0, 10);
				net.setTarget(0, res, -1, 1);
				net.run(); 
				net.learn();
			}
			
			net.applyWeightDeltas(); 
				
			//System.out.println("Out1:" + net.getOutput(0) + " Out2:" + net.getOutput(1));

			if (every(i,100)) {
				net.run();
				display.cls(Color.ORANGE);
				int y=0;
				double error=0;
				for (int x=0;x<300;x++) {
					net.setInput(0, (double)x/50.0, 0, 10);
					net.run();
					net.setTarget(0, function((double)x/50.0), -1, 1);
					error+=net.errorTotal;
					y = transformGraphValue(0);
					display.setDrawColor(Color.gray);
					display.drawRect(x, y, x+2, y+2);
					y = transformGraphValue(function((double)x/50.0));
					display.setDrawColor(Color.BLUE);
					display.drawRect(x, y, x+2, y+2);
					y = transformGraphValue(net.getOutput(0,-1,1));
					display.setDrawColor(Color.green);
					display.drawRect(x, y, x+2, y+2);
				}
				net.drawNetwork(display, 330, 20, 300, 200);
				graphError.addData(error/300.0);
				//graphError.draw(display, 20, 450, 350, 100, Color.gray);
				graphError.draw(display, 20, 200, 300, 200, Color.gray);
				display.refresh();
				
				for (int j=0; j<net.numConnections; j++) {
					int w = 200+(int)(net.getWeight(j)*25.0);
					graphError.addData(w/30.0);
					display.setDrawColor(Color.BLUE);
					display.drawFilledRect((int)(w), 120+j, 2, 2);
				}
				
				System.out.println("Iterations:" + i + " \tError: " + error/300.0);
				//net.learningRate = Math.abs(error)/300.0;
			}
		}
	}
	
	public static double function(double x) {
		//return (x-1)/5;
		//return Math.tanh(x);
		double val = (Math.sin(x*1)*0.4 + (Math.sin(x*2)*0.4));//+ (Math.sin(x*5)*0.2));
		//return (Math.cos(x)*0.5)+0.5;
		//if (val<0) val=-0.5;
		//if (val>0) val=0.5;
		return val;
	}
	
	public static int transformGraphValue(double val) {
		val = (val+1.0)*50.0;
		int ret = (int)val;
		if (ret<0) ret=5;
		if (ret>200) ret=195;
		ret+=25;
		return ret;
	}
	
	public static boolean every(int x, int i) {
		if (x==0) return true;
		if (x%i==0) return true;
		return false;
	}
}

