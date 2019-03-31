package com.physmo.neural.examples;

import com.physmo.neural.activations.ActivationType;
import com.physmo.neural.NN2;
import com.physmo.neural.NeuralNet;
import com.physmo.minvio.BasicDisplay;
import com.physmo.minvio.BasicDisplayAwt;
import com.physmo.minvio.BasicGraph;

import java.awt.*;

public class TestNeuralNet {

	public static void main(String[] args) {
		// [*] testBasic();

		//testSin();
		//testSinNN2();
		//testMapping();
		
//		TestBinaryClassifier testBC = new TestBinaryClassifier();
//		testBC.run();
		
		//TestAudio testAudio = new TestAudio();
		//testAudio.run();
		
		//TestRecurrent testRecurrent = new TestRecurrent();
		//testRecurrent.run();
		
//		TestRecurrentNN2 testRecurrent = new TestRecurrentNN2();
//		testRecurrent.run();
	}

	public static void testMapping() {
		NeuralNet net = new NeuralNet();
		System.out.println("mapValue(-1,-1,1): " + net.mapValue(-1, -1, 1));
		System.out.println("mapValue(1,-1,1): " + net.mapValue(1, -1, 1));
		System.out.println("unmapValue(0.1,-1,1): " + net.unmapValue(0.1, -1, 1));
		System.out.println("unmapValue(0.9,-1,1): " + net.unmapValue(0.9, -1, 1));
	}
	
//	public static void testBasic() {
//		NeuralNet net = new NeuralNet();
//		net.buildNet("1 30 1");
//		for (int i=0;i<5000;i++) {
//			net.setInput(0, 0, 0, 1);
//			//net.setInput(1, -1);
//			net.setTarget(0, 0.1234, 0, 1);
//			//net.setTarget(1, 0.0);
//			net.run();
//			net.calculateLearningDeltas();
//			//System.out.println("Out1:" + net.getOutput(0) + " Out2:" + net.getOutput(1));
//			net.run();
//			if (every(i,1000)) System.out.println("Error: " + net.errorTotal + "output:" + net.getOutput(0, 0, 1));
//		}
//	}
	
	public static void testSin() {
		BasicDisplay display = new BasicDisplayAwt(640, 480);
		BasicGraph graphError = new BasicGraph(20000);
		NeuralNet net = new NeuralNet();
		net.buildNet("1 4 1");
		net.randomiseAllWeights(-2, 2);
		net.learningRate = 0.0015;
		net.momentum = 0.65;
		display.startTimer();
		
		for (int i=0;i<50000000;i++) {
			for (int e=0;e<10;e++) {
				double rnd = Math.random()*9.00;
				double res = function(rnd);
				net.setInput(0, rnd, 0, 10);
				net.setTarget(0, res, -1, 1);
				net.run(); 
				net.calculateLearningDeltas();
			}
			
			net.applyWeightDeltas(); 
				
			//System.out.println("Out1:" + net.getOutput(0) + " Out2:" + net.getOutput(1));
			

			//if (every(i,100)) {
			if (display.getEllapsedTime()>1000/30)
			{
				display.startTimer();
				
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
				graphError.draw(display, 20, 170, 300, 300, Color.gray);
				
				
				for (int j=0; j<net.numConnections; j++) {
					int w = 200+(int)(net.getWeight(j)*25.0);
					graphError.addData(w/30.0);
					display.setDrawColor(Color.BLUE);
					display.drawFilledRect((int)(w), 120+j, 2, 2);
				}
				
				//System.out.println("Iterations:" + i + " \tError: " + error/300.0);
				display.refresh();
				//net.learningRate = Math.abs(error)/300.0;
			}
		}
	}
	
	public static void testSinNN2() {
		BasicDisplay display = new BasicDisplayAwt(640, 480);
		BasicGraph graphError = new BasicGraph(2000);
		NN2 net = new NN2()
				.addLayer(1,ActivationType.TANH)
				.activationType(ActivationType.TANH)
				.addLayer(15,ActivationType.TANH)
				.activationType(ActivationType.TANH)
				.addLayer(15,ActivationType.TANH)
				.activationType(ActivationType.TANH)
				.addLayer(1,ActivationType.TANH)
				.activationType(ActivationType.TANH)
				.randomizeWeights(-0.2, 0.2)
				.inputMapping(1, 0)
				.outputMapping(1, 0)
				.learningRate(0.015)
				.dampenValue(0.5);
				
//		net.buildNet("1 4 1");
//		net.randomiseAllWeights(-2, 2);
//		net.learningRate = 0.0015;
//		net.momentum = 0.65;
		display.startTimer();
		
		for (int i=0;i<50000000;i++) {
			for (int e=0;e<10;e++) {
				double rnd = Math.random()*6.00;
				double res = function(rnd);
				
				net.setInputValue(0, rnd);
				net.setOutputTargetValue(0, res);
				net.feedForward();
				net.backpropogate();
				//net.calculateLearningDeltas();
			}
			
			net.learn();
			
			//net.applyWeightDeltas(); 
				
			//System.out.println("Out1:" + net.getOutput(0) + " Out2:" + net.getOutput(1));
			

			//if (every(i,100)) {
			if (display.getEllapsedTime()>1000/30)
			{
				display.startTimer();
				
				net.run(false);
				display.cls(Color.ORANGE);
				int y=0;
				double error=0;
				for (int x=0;x<300;x++) {
					net.setInputValue(0, (double)x/50.0);
					net.setOutputTargetValue(0, function((double)x/50.0));
					//net.run(false);
					net.feedForward();
					
					error+=net.getCombinedError();
					y = transformGraphValue(0);
					display.setDrawColor(Color.gray);
					display.drawRect(x, y, x+2, y+2);
					y = transformGraphValue(function((double)x/50.0));
					display.setDrawColor(Color.BLUE);
					display.drawRect(x, y, x+2, y+2);
					y = transformGraphValue(net.getOutputValue(0));
					display.setDrawColor(Color.green);
					display.drawRect(x, y, x+2, y+2);
				}
				
				// fixme
//				net.drawNetwork(display, 330, 20, 300, 200);
				graphError.addData(error/300.0);
				graphError.draw(display, 20, 170, 300, 300, Color.gray);
				
				
				double newLearningRate = (Math.abs(error)/300.0)*0.1;
				if (newLearningRate>0.01) newLearningRate=0.01;
				net.learningRate(newLearningRate);
				
//				for (int j=0; j<net.numConnections; j++) {
//					int w = 200+(int)(net.getWeight(j)*25.0);
//					graphError.addData(w/30.0);
//					display.setDrawColor(Color.BLUE);
//					display.drawFilledRect((int)(w), 120+j, 2, 2);
//				}
				
				//System.out.println("Iterations:" + i + " \tError: " + error/300.0);
				display.refresh();
				//net.learningRate = Math.abs(error)/300.0;
			}
		}
	}
	public static double function(double x) {
		//return (x-1)/5;
		//return Math.tanh(x);
		double val = (Math.sin(x*1)*0.4 + (Math.sin(x*2)*0.4)) + (Math.sin(x*5)*0.2);
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

