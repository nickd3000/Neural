package testNeuralNet;

import Neural.NeuralNet;
import ToolBox.BasicDisplay;
import ToolBox.BasicGraph;

import javax.sound.sampled.AudioFormat;
import javax.sound.sampled.AudioSystem;
import javax.sound.sampled.SourceDataLine;
import java.awt.*;
import java.io.*;

public class TestAudio {

	NeuralNet net = new NeuralNet();
	byte[] fileData;
	int fileLength = 0;
	BasicDisplay display = new BasicDisplay(300,200); 
	BasicGraph graphError = new BasicGraph(200);
	BasicGraph graphGenerated = new BasicGraph(1000);
	BasicGraph graphInput = new BasicGraph(100);
	BasicGraph graphOutput = new BasicGraph(100);
	int sampleLength=40000;
	byte[] genSample = new byte[sampleLength];
	int numInputs = 100;
	
	public void run() {
		// atanh(x) = (log(1+x) - log(1-x))/2
		readDataFile();
		
		
		net.buildNet("100 20 2 20 100");
		net.learningRate=0.013;
		net.momentum=0.65;
	
		
		double error = 0;
		int batch=0;
		int batchSize=5;
		for (int i=0;i<10000;i++) {
				error=0;
			for (int j=0;j<fileLength/2;j++) {
				batch++;
				insertTrainingData(numInputs);
				net.run();
				error += net.errorTotal;
				net.learn();
				
				if (batch>batchSize) {
					batch=0;
					net.applyWeightDeltas();
				}
			}
			
			error /= (fileLength/2);
			graphError.addData(error*2.0);

			display.cls(Color.gray);
			//graphError.draw(display, 450, 350, 320, 40, Color.yellow);
			generateAutoencoder();
			graphError.draw(display, 10, 100, 200,10, Color.green);
			graphInput.draw(display, 10, 50, 200,100, Color.white);
			graphOutput.draw(display, 10, 50, 200,100, Color.black);
			display.setDrawColor(Color.WHITE);
			display.drawText("Iteration  "+i, 40, 40);
			display.drawText("Error  "+error, 40, 65);
			
			
			//if (true || display.getMousePosition()!=null) {
				//generate();
				//generateAutoencoder();
				//graphGenerated.draw(display, 10, 100, 200,50, Color.orange);
			//}
			
			display.refresh();
			
			//if (display.getMousePosition()!=null) playSound();
			
		}	
			
	}
	
	public void generateAutoencoder() {
		
		int samplePosition = (int)(Math.random()*(fileLength-numInputs));
		for (int i=0;i<numInputs;i++) {
			double sample = getSamplePoint(samplePosition+i);
			net.setInput(i, sample, -1, 1);
			net.setTarget(i, sample, -1, 1);
			graphInput.addData(sample);
		}
		net.run();
		for (int i=0;i<numInputs;i++) {
			graphOutput.addData(net.getOutput(i, -1, 1));
		}
	}
	
	public void generate() {
		double jitterScale = 0.01;
		double [] tempGen = new double[sampleLength];
		for (int i=1;i<numInputs;i++) {
			tempGen[i]=(tempGen[i-1]+((Math.random()-0.5)*0.1))/2.0;
		}
		for (int i=0;i<sampleLength-numInputs;i++) {
			for (int j=0;j<numInputs;j++) {
				double jitter = (Math.random()-0.5)*jitterScale;
				net.setInput(j, tempGen[i+j]+jitter, -1, 1);
			}
			net.run();
			double result = net.getOutput(0, -1, 1);

			tempGen[i+numInputs]=result;
			genSample[i+numInputs] = (byte) convertDoubleToByte(result);
			
			// Test
			//genSample[i+numInputs] = (byte) convertDoubleToByte(getSamplePoint(i+numInputs));
			
			graphGenerated.addData(result);
		}
	}
	
	public void playSound() {
		try {
			final AudioFormat audioFormat = new AudioFormat((float)(16000.0*2.8), 8, 1, false, true);
	        SourceDataLine line = AudioSystem.getSourceDataLine(audioFormat );
	        line.open(audioFormat );
	        line.start();
	        
	        for (int i = 0; i < 3; i++) {//repeat in loop
	        	
	            //play(line, genSample);
	        		line.write(genSample, 0, genSample.length);
	        }
	        line.drain();
	        line.close();
	    } catch (Exception e) {
	        e.printStackTrace();
	    }
	}
	
	public void insertTrainingData(int numInputs) {
		
		int samplePosition = (int)(Math.random()*(fileLength-numInputs));
		for (int i=0;i<numInputs;i++) {
			double sample = getSamplePoint(samplePosition+i);
			net.setInput(i, sample, -1, 1);
			net.setTarget(i, sample, -1, 1);
		}
		//net.setTarget(0, getSamplePoint(samplePosition+numInputs), -1, 1);
	}
	
	public double getSamplePoint(int pos) {
		double amplify = 3.0;
		int v =  fileData[pos];
		v = ~v & 0xff;
		return ((v/128.0)-1)*amplify;
	}
	
	public int convertDoubleToByte(double v) {
		double amplify = 3.0;
		v/=amplify;
		v+=1;
		v*=128;
		int i = (int)v;
		if (i<0) i=0;
		if (i>255) i=255;
		i = ~i & 0xff;
		return i;
	}
	
	public void readDataFile() {
		File file = new File("raw_u8_pcm_01.raw");
		fileLength = (int)file.length();
	    fileData = new byte[(int) file.length()];
	    DataInputStream dis = null;
		try {
			dis = new DataInputStream(new FileInputStream(file));
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	    try {
			dis.readFully(fileData);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	    try {
			dis.close();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
}
