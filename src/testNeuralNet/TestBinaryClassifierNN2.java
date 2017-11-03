package testNeuralNet;


import java.awt.*;

import com.physmo.toolbox.BasicDisplay;
import com.physmo.toolbox.BasicGraph;

import Activations.ActivationType;
import Neural.NN2;
import Neural.NeuralNet;

public class TestBinaryClassifierNN2 {

	public static void main(String[] args) {
		TestBinaryClassifierNN2 bc = new TestBinaryClassifierNN2();
		bc.run();
	}
	
	enum Shape {SQUARE, HALF, SQUAREHOLE, BLOBS, RINGS};
	class DataPoint {
		public double x,y,v;
	}
	
	Color colDataPointOn = new Color(0xFA1FFF);
	Color colDataPointOff = new Color(0x9AA899);
	Color colMatrixOn = new Color(0x4A7B9D);
	Color colMatrixOff = new Color(0x54577C);
	Color colMatrixMid = new Color(0x44476C);
	
	static int numPoints = 2000;
	DataPoint [] data = new DataPoint[numPoints]; 
	NN2 net = null;
	
	BasicDisplay display = new BasicDisplay(800, 600);
	

	public void run() {
		
		net = new NN2()
				.addLayer(2)
				.activationType(ActivationType.TANH)
				.addLayer(10)
				.activationType(ActivationType.TANH)
				.addLayer(10)
				.activationType(ActivationType.TANH)
				.addLayer(1)
				.activationType(ActivationType.TANH)
				.learningRate(0.013)
				.dampenValue(0.1951)
				.randomizeWeights(-0.1, 0.1)
				.inputMapping(1, 0)
				.outputMapping(1, 0);


		BasicGraph graphError = new BasicGraph(2000);

//		net.buildNet("2 2 2 1");
//		net.learningRate=0.00013;
//		net.momentum=0.45;
		initData(Shape.BLOBS);
		int batch=0;
		int batchSize=100;

		double error=0.0;
		for (int i=0;i<100000;i++) {
			
			for (int j=0;j<numPoints;j++) {
				batch++;
				int d = (int)(Math.random()*numPoints);
				net.setInputValue(0, data[d].x);// -1, 1);
				net.setInputValue(1, data[d].y);//, -1, 1);
				net.setOutputTargetValue(0, data[d].v);//, -1, 1);
				
				//net.run(true);
				
				net.feedForward();
				net.backpropogate();
				//net.backpropogate();
				//net.learn();
				
				if (batch>batchSize) {
					batch=0;
					
					net.learn();
				}
				
			}
			
			error=0.0;
			for (int j=0;j<numPoints;j++) {
				net.setInputValue(0, data[j].x);//, -1, 1);
				net.setInputValue(1, data[j].y);//, -1, 1);
				net.setOutputTargetValue(0, data[j].v);//, -1, 1);
				//net.run(false);
				net.feedForward();
				error += net.getCombinedError();
			}
			
			error = (error/(double)numPoints);
			//if (error>0.0001) net.learningRate( Math.abs(error) * 0.01);
			//if (net.learningRate>0.01) net.learningRate=0.01;
			//if (net.learningRate<0.0000001) net.learningRate=0.0000001;
	
			graphError.addData(error*10.0);

			if (i%10==0) {
				display.cls(new Color(100,200,100));
				display.setDrawColor(Color.BLACK);
				
				graphError.draw(display, 10, 420, 400, 120, Color.yellow);
				
				drawMatrix(10, 10, 200);
				drawPoints(10, 10, 200);
				//net.drawNetwork(display, 430, 10, 300, 300);
				
				display.setDrawColor(Color.WHITE);
				display.drawText("Iteration  "+i, 10, 555);
				display.drawText("Error  "+error, 10, 570);
				//display.drawText("Learning rate  "+net.learningRate, 10, 585);
		
				display.refresh();
			}
		}
	}
	
	
	
	public void initData(Shape shape) {
		double x,y;
		for (int i=0;i<numPoints;i++) {
			x = (Math.random()-0.5)*2.0;
			y = (Math.random()-0.5)*2.0;
			data[i] = new DataPoint();
			data[i].x=x;
			data[i].y=y;
			data[i].v=rule(shape,x,y);
		}
	}
	
	public double rule(Shape shape, double x, double y) {
		double val=-1;
		switch (shape) {
		case HALF:
			val = x<0?0:1;
			break;
		case SQUARE:
			val = -1;
			if (x>-0.5 && x<0.5 && y>-0.5 && y<0.5) val=1;
			break;
		case SQUAREHOLE:
			val = -1;
			if (x>-0.5 && x<0.5 && y>-0.5 && y<0.5) val=1;
			if (x>-0.25 && x<0.25 && y>-0.25 && y<0.25) val=-1;
			break;
		case BLOBS:
			if (isPointInsideCircle(x, y, -0.2, -0.2, 0.2)) return 1;
			if (isPointInsideCircle(x, y, 0.4, 0.5, 0.3)) return 1;
			if (isPointInsideCircle(x, y, -0.7, 0.7, 0.3)) return 1;
			if (isPointInsideCircle(x, y, 0.7, -0.9, 0.3)) return 1;
			return -1;
		case RINGS:
			double d=Math.sqrt((x*x)+(y*y));
			int dd = (int)(d*6.0);
			if ((dd&1)==1) return -1;
			else return 1;
			
		}
		return val;
	}
	
	public boolean isPointInsideCircle(double x, double y, double cx, double cy, double r) {
		double d = Math.sqrt(((x-cx)*(x-cx))+((y-cy)*(y-cy)));
		if (d<r) return true;
		return false;
	}
	
	public void drawPoints(int xo, int yo, int scale) {
		double x,y,v;
		Color c;
		for (int i=0;i<numPoints;i++) {
			x = (data[i].x + 1.0) * (double)scale;
			y = (data[i].y + 1.0) * (double)scale;
			v = data[i].v;
			if (v<0.2) c = colDataPointOff;
			else if (v>0.8) c = colDataPointOn;
			else c = Color.lightGray;
			display.setDrawColor(c);
			display.drawCircle(x+xo, y+yo, 5);
		}
	}
	
	public Color fuck(double v) {
		int vv =(int) ((v+0.5)*200.0);
		if (vv<0) vv=0;
		
		if (vv>255) vv=255;
		return new Color(50,50,vv);
	}
	
	public void drawMatrix(int xo, int yo, int scale) {
		double xx,yy,v;
		Color c;
		int sections=50;
		int step =200 / 50;
		double drawScale = (double)scale / 100.0;
		
		for (int y=-100;y<100;y+=step) {
			for (int x=-100;x<100;x+=step) {
				xx = x * 0.01;//(data[i].x + 1.0) * (double)scale;
				yy = y * 0.01;//y = (data[i].y + 1.0) * (double)scale;
				net.setInputValue(0, xx);//, -1, 1);
				net.setInputValue(1, yy);//, -1, 1);
				
				//net.run(false);
				net.feedForward();
				
				v = net.getOutputValue(0);//, -1, 1);
				if (v<0.3) c = colMatrixOff;
				else if (v>0.7) c = colMatrixOn;
				else c = colMatrixMid;
				//display.drawCircle(x, y, 5, c);
				int dx = (int)((x + 100) * drawScale);
				int dy = (int)((y + 100) * drawScale);
				c = fuck(v);
				display.setDrawColor(c);
				display.drawFilledRect(dx+xo,dy+yo,(int)(step*drawScale),(int)(step*drawScale));
			}
		}
	}
}
