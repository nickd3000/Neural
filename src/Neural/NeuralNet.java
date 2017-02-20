package Neural;

import ToolBox.BasicDisplay;
import ToolBox.LookupTable;

import java.awt.*;
import java.util.ArrayList;
import java.util.Scanner;
import java.util.function.DoubleUnaryOperator;

//import java.util.List;


// http://www.ece.utep.edu/research/webfuzzy/docs/kk-thesis/kk-thesis-html/node73.html
// http://www.cse.unsw.edu.au/~cs9417ml/MLP2/
// http://www.doc.ic.ac.uk/~sgc/teaching/pre2012/v231/lecture13.html

class Link {
	int layerId;
	int sourceNodeId;
	int targetNodeId;
	
	public double weight;

	double delta; // Accumulated training deltas.
	
	Link() {
		weight=delta=0; }
	
	public void set(int layerId, int sourceNodeId, int targetNodeId) {
		this.layerId = layerId;
		this.sourceNodeId = sourceNodeId;
		this.targetNodeId = targetNodeId;
	}
}

class Node {
	double value;
	double sum;
	double error;
	int layerId;
	int nodeSequence; // Index of node in this layer.
	Node() {
		value=sum=error=0;
	}
}


public class NeuralNet {

	public LookupTable lookupTanH;
	public LookupTable lookupDerivative;
	
	public double learningRate = 0.025;
	public double momentum = 0.4;
	
	public Link [] links;
	Node [] nodes;
	double [] targetValues;
	
	public int numConnections = 0;
	int numNodes = 0;
	int numLayers = 0;
	ArrayList <Integer> layerSizes = new ArrayList<>();
	
	// Index of the first node in each layer.
	ArrayList <Integer> nodeOffsets = new ArrayList<>();
	ArrayList <Integer> weightOffsets = new ArrayList<>();
	
	//double mapperMin = 0;
	//double mapperMax = 1;
	
	public double errorTotal = 0;
	
	public NeuralNet() {
	}
		
	public void buildNet(String structure) {
		
		// Init maths lookup tables.
		DoubleUnaryOperator dl = (x) -> {return Math.tanh(x);};
		lookupTanH = new LookupTable(-10.0,10.0,5000, dl);

		DoubleUnaryOperator dl2 = (x) -> {return (1-(x*x));};
		lookupDerivative = new LookupTable(-2.0,2.0,5000, dl2);
		
		
		/*
		long time1 = System.nanoTime();
		long time2 = System.nanoTime();
		int numTests = 1000000;
		double v = 0;
		time1 = System.nanoTime();
		for (int i=0;i<numTests;i++) {
			v += Math.tanh(i*0.000001);
		}
		time2 = System.nanoTime();
		System.out.println("Tanh:   " + (time2-time1));
		v = 0;
		time1 = System.nanoTime();
		for (int i=0;i<numTests;i++) {
			v += lookupTanH.getValue(i*0.000001);
		}
		time2 = System.nanoTime();
		System.out.println("lookup: " + (time2-time1));
		*/
		
		
		Scanner scanner = new Scanner(structure);
		while (scanner.hasNextInt()) {
			numLayers++;
			int layerSize = scanner.nextInt();
			layerSizes.add( layerSize );
			System.out.println("Builder layer size:"+layerSize+"  numLayers:"+numLayers);
		}
		
		// Add a bias node to all but last layer.
		for (int i=0;i<numLayers-1;i++) {
			layerSizes.set(i, layerSizes.get(i)+1);
		}
		
		/*
		numLayers = 3;
		layerSizes.add(2+1);
		layerSizes.add(2+1);
		layerSizes.add(2+1);
		*/
		
		
		for (int i =0; i<numLayers;i++) {
			nodeOffsets.add(numNodes);
			numNodes+=layerSizes.get(i);
			if (i>0) {
				numConnections+=layerSizes.get(i-1) * layerSizes.get(i);
			}
			weightOffsets.add(numConnections);
		}
		
		System.out.println(toString());
	
		links = new Link [numConnections];
		nodes = new Node [numNodes];
		targetValues = new double [layerSizes.get(numLayers-1)];
		
		for (int i=0;i<numNodes;i++) {
			nodes[i] = new Node();
		}
		
		// Link the connections.
		int linkId = 0;
		for (int layerId = 0; layerId<numLayers-1; layerId++) {
			int thisLayerNodeCount = layerSizes.get(layerId);
			int nextLayerNodeCount = layerSizes.get(layerId+1);
			
			for (int thisNodeId = 0; thisNodeId<thisLayerNodeCount; thisNodeId++) {
				
				for (int nextNodeId = 0; nextNodeId<nextLayerNodeCount; nextNodeId++) {
					
					links[linkId] = new Link();
					links[linkId].layerId = layerId;
					links[linkId].sourceNodeId = thisNodeId+nodeOffsets.get(layerId);
					links[linkId].targetNodeId = nextNodeId+nodeOffsets.get(layerId+1);
					
					System.out.println("linkId:"+linkId+" layer:"+layerId+" src:"+links[linkId].sourceNodeId+" dst:"+links[linkId].targetNodeId);
					
					linkId++;
				}
			}
		}
		
		for (int layerId = 0; layerId<numLayers; layerId++) {
			int thisLayerNodeCount = layerSizes.get(layerId);
			
			for (int thisNodeId = 0; thisNodeId<thisLayerNodeCount; thisNodeId++) {
				nodes[thisNodeId+nodeOffsets.get(layerId)].layerId = layerId;
				nodes[thisNodeId+nodeOffsets.get(layerId)].nodeSequence = thisNodeId;
			}
		}
		
		randomiseAllWeights(-0.3, 0.3);
	}
	
	static double activationMax = 0;
	static double activationMin = 0;
	
	
	public double activation(double val) {
		//return (1 / (1 + Math.exp(-val)));
		//if (val<activationMin) activationMin=val;
		//if (val>activationMax) activationMax=val;
		
		//double ret = Math.tanh(val);
		double ret = lookupTanH.getValue(val);
		
		//validateDouble(ret);
		return ret;
	}
	
	
	// Inputs to this seem to be -1 to 1.
	public double derivative(double val) {
		//double ret=1-(val*val);

		double ret = lookupDerivative.getValue(val);
		
		return ret;
	}
	
	public double mapValue(double val, double min, double max) {
		
		//double targetMin = 0.1;
		//double targetMax = 0.9;
		double targetMin = -0.8;
		double targetMax = 0.8;
		
		if (val<min) val=min;
		if (val>max) val=max;
		val=(val-min)/(max-min) * (targetMax-targetMin) + targetMin;
		
		//validateDouble(val);
		
		return val;
	}
	public double unmapValue(double val, double min, double max) {
		//double targetMin = 0.1;
		//double targetMax = 0.9;
		double targetMin = -0.8;
		double targetMax = 0.8;
		
		if (val<targetMin) val=targetMin;
		if (val>targetMax) val=targetMax;
		val=(val-targetMin)/(targetMax-targetMin) * (max-min) + min;
		
		//validateDouble(val);
		
		return val;
	}
	
	public void setInput(int i, double value, double min, double max) {
		nodes[i].value = mapValue(value,min,max);
	}
	
	public void setTarget(int i, double value, double min, double max) {
		targetValues[i] = mapValue(value,min,max);
	}
	
	public double getOutput(int i, double min, double max) {
		int offset = nodeOffsets.get(numLayers-1);
		return unmapValue(nodes[offset+i].value,min,max);
	}
	

	
	// temp name
	public void run () {
		clearValues(true);
		setBiasNodes();
		propogateForward();
		errorTotal = calculateOutputError();
	}
	
	public void propogateForward() {
		
		//System.out.println("propogateForward:");
		
		int thisLayerNodeCount;
		int nextLayerNodeCount;
		int numWeights;
		int weightsOffset;
		double sourceValue;
		double weight;
	
		
		for (int layerId=0;layerId<numLayers-1;layerId++) {
			
			thisLayerNodeCount = layerSizes.get(layerId);
			nextLayerNodeCount = layerSizes.get(layerId+1);
			numWeights = thisLayerNodeCount * nextLayerNodeCount;
			weightsOffset = weightOffsets.get(layerId);
			
			//System.out.println(" Layer: "+layerId);
			
			for (int weightId = weightsOffset; weightId<weightsOffset+numWeights;weightId++) {
				//System.out.println("   Processing weightId: "+weightId);
				if (links[weightId].layerId!=layerId) System.out.println("Link in wrong layer.");
				
				Link link = links[weightId];
				sourceValue = nodes[link.sourceNodeId].value;
				weight = link.weight;
				nodes[link.targetNodeId].sum += sourceValue * weight;
				
				//validateDouble(nodes[link.targetNodeId].sum);
				
				//System.out.println("    " + sourceValue + " " + weight + "  " );
			}
			
			activateLayer(layerId+1);
		}
		
	}
	

	public double calculateOutputError() {
		/*
			SignalError = (ExpectedOutput - Output) * Output * (1-Output);
				 error = o1(1 - o1)(t1 - o1)
		 */
		int numOutputNodes = layerSizes.get(numLayers-1);
		int nodeOffset = nodeOffsets.get(numLayers-1);
		double totalError = 0;
		for (int i=0;i<numOutputNodes;i++) {
			double output = nodes[nodeOffset+i].value;
			double target = targetValues[i];
			double diff = (target-output);

			//double error = diff * output * (1.0-output);
			double error = diff * derivative(output);

			//validateDouble(error);
			
			nodes[nodeOffset+i].error = error;
			totalError+=Math.abs(error);
			//totalError+=(error);
		}
		
		return totalError;
	}
	
	
	public void learn() {
		/*
		 * // Calculate weight difference between node j and k
					Layer[i].Node[j].WeightDiff[k] = 
						LearningRate * 
						Layer[i].Node[j].SignalError*Layer[i-1].Node[k].Output +
						Momentum*Layer[i].Node[j].WeightDiff[k];
		 */
		
		
		for (int layerId = numLayers-2;layerId>=0;layerId--) {
			int numWeights = layerSizes.get(layerId) * layerSizes.get(layerId+1);
			int weightOffset = weightOffsets.get(layerId);
			
			for (int n = 0; n<layerSizes.get(layerId);n++) {
				int nodeId = n + nodeOffsets.get(layerId);
				nodes[nodeId].error=0;
			}
				
			for (int w = weightOffset;w<weightOffset+numWeights;w++) {
				Link link = links[w];
				double targetError = nodes[link.targetNodeId].error;
				//validateDouble(targetError);
				nodes[link.sourceNodeId].error += targetError * link.weight;
				
				double change = targetError * nodes[link.sourceNodeId].value;
				//link.weight += learningRate * change;  
				link.delta += learningRate * change;
				//validateDouble(link.delta);
				//validateDouble(change);
						
			}
			
			// Second step
			// error = ratio * output * (1.0 - output) * sum;
			for (int n = 0; n<layerSizes.get(layerId);n++) {
				int nodeId = n + nodeOffsets.get(layerId);
				double combinedError = nodes[nodeId].error;
				//double modifiedError = combinedError;
				//double modifiedError = nodes[nodeId].value * (1.0 - nodes[nodeId].value) * combinedError;
				double modifiedError = derivative(nodes[nodeId].value) * combinedError;
				//validateDouble(modifiedError);
				nodes[nodeId].error = modifiedError;
			}
			
		}
		
	}
	
	public void applyWeightDeltas() {
		for (int i=0;i<numConnections;i++) {
			
			//if (Math.random()>0.5) {
				links[i].weight += links[i].delta;
			
			links[i].delta*=momentum;	// Reduce the delta, which creates a momentum effect.
			//validateDouble(links[i].delta);
			//}
		}
	}
	
	// Apply activation function to all nodes in layer.
	public void activateLayer(int layerId) {
		int numNodes = layerSizes.get(layerId);
		int nodeOffset = nodeOffsets.get(layerId);
		Node n;
		
		for (int i=nodeOffset;i<nodeOffset+numNodes;i++) {
			n = nodes[i];
			//n.value = Math.tanh(n.sum);
			n.value = activation(n.sum);
			
			// Set bias node to 1.
			if (layerId<numLayers-1 && i==nodeOffset+numNodes-1) n.value=1.0;
		}
		
		// Experimental sparsify
		double maxValue=0;
		int maxId=0;
		if (layerId==2000) // disable this.
		{
			for (int i=nodeOffset;i<nodeOffset+numNodes;i++) {
				n = nodes[i];
				if (n.value>maxValue) {
					maxValue=n.value;
					maxId=i;
				}
			}
			for (int i=nodeOffset;i<nodeOffset+numNodes;i++) {
				if (i!=maxId) nodes[i].value=-0.8;
				else nodes[i].value=0.8;
			}
		}
	}
	
	public static double _sigmoid(double x)
	{
	    return (1 / (1 + Math.exp(-x)));
	}
	
	public void randomiseAllWeights(double min, double max) {
		double span=max-min;
		for (Link l : links) {
			l.weight = min + (Math.random()*span);
			//l.weight = 0.1;
		}
	}
	
	
	// Clears values and sets bias values.
	public void clearValues(boolean skipFirstLayer) {
		
		for (int l = 0; l<numLayers;l++) {
			if (l==0 && skipFirstLayer==true) continue;
			
			int numNodesInLayer = layerSizes.get(l);
			int nodeOffset = nodeOffsets.get(l);
			for (int n = 0;n<numNodesInLayer;n++) {
				nodes[nodeOffset+n].value = 0.0;
				nodes[nodeOffset+n].sum = 0.0;
				nodes[nodeOffset+n].error = 0.0;
			}
		}
		
	}
	
	public void setBiasNodes() {
		
		for (int l = 0; l<numLayers-1;l++) {
			int numNodesInLayer = layerSizes.get(l);
			int nodeOffset = nodeOffsets.get(l);

			// Set bias node to 1.0	
			nodes[nodeOffset+numNodesInLayer-1].value=1.0;
			nodes[nodeOffset+numNodesInLayer-1].sum=1.0;
		
		}
		
	}

	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append("NETWORK STRUCTURE \n" );
		sb.append("numLayers:" + numLayers + "\n");
		sb.append("layerSizes:" );
		for (int s : layerSizes) {
			sb.append(""+s+", ");
		}
		sb.append("\n");
		
		sb.append("nodeOffsets:" );
		for (int s : nodeOffsets) {
			sb.append(""+s+", ");
		}
		sb.append("\n");
		
		sb.append("weightOffsets:" );
		for (int s : weightOffsets) {
			sb.append(""+s+", ");
		}
		sb.append("\n");
		
		sb.append("numNodes:" + numNodes + "\n");
		sb.append("numConnections:" + numConnections + "\n");
		
		return sb.toString();
	}
	
	public void drawNetwork(BasicDisplay g, int xo, int yo, int width, int height) {
		int xdiff=width;
		int ydiff=height;
		
		//g.setDrawColor(Color.black);
		//g.drawRect(xo,yo,xo+width,yo+height);
		
		for (int i=0;i<numConnections;i++) {
			int id1 = links[i].sourceNodeId;
			int id2 = links[i].targetNodeId;
			int x1 = xo+getNodeX(nodes[id1].layerId,nodes[id1].nodeSequence,xdiff,ydiff);
			int y1 = yo+getNodeY(nodes[id1].layerId,nodes[id1].nodeSequence,xdiff,ydiff);
			int x2 = xo+getNodeX(nodes[id2].layerId,nodes[id2].nodeSequence,xdiff,ydiff);
			int y2 = yo+getNodeY(nodes[id2].layerId,nodes[id2].nodeSequence,xdiff,ydiff);
			//g.setDrawColor(getColorForValue(links[i].weight));
			//g.drawLine(x1, y1, x2, y2);
			drawWeight(g,x1,y1,x2,y2,links[i].weight);
		}
		for (int i=0;i<numNodes;i++) {
			int nx = xo+getNodeX(nodes[i].layerId,nodes[i].nodeSequence,xdiff,ydiff);
			int ny = yo+getNodeY(nodes[i].layerId,nodes[i].nodeSequence,xdiff,ydiff);
			g.setDrawColor(getColorForValue(nodes[i].value));
			//g.drawFilledCircle(nx, ny, 10);
		}
	}
	public void drawWeight(BasicDisplay g,int x1,int y1, int x2, int y2, double weight) {
		g.setDrawColor(Color.BLACK);
		if (weight<-1) weight=-1;
		if (weight>1) weight=1;
		double p = (weight + 1.0f)/2.0f;
		g.drawFilledRect(
				x1+(int)((double)(x2-x1)*p)-2,
				y1+(int)((double)(y2-y1)*p)-2,4,4);
		
		g.setDrawColor(getColorForValue(weight));
		g.drawLine(x1, y1, x2, y2);
	}
	public Color getColorForValue(double val) {
		
		//int ival = (int)(val*255);
		int ival = (int)(val*128)+128;
		
		if (ival<0) ival=0;
		if (ival>255) ival=255;
		
		Color col = new Color(ival,ival,ival);
		return col;
	}
	public int getNodeX(int layerId, int nodeSequence, int xdiff, int ydiff) {
		
		return (int)(((float)xdiff / (float)(numLayers-1)) * (float)layerId);
		
		//return layerId*xdiff;
	}
	public int getNodeY(int layerId, int nodeSequence, int xdiff, int ydiff) {
		int numNodesInLayer = layerSizes.get(layerId);
		return (int)(((float)ydiff / (float)(numNodesInLayer-1)) * (float)nodeSequence);
		//return nodeSequence*ydiff;
	}
	
	// Used for experimentation.
	public void setWeight(int id, double val) {
		if (id<=0) return;
		if (id>=numConnections) return;
		if (val<0) val=0;
		links[id].weight = val;
	}
	public double getWeight(int id) {
		if (id<0) return 0.0;
		if (id>=numConnections) return 0.0;
		return links[id].weight;
	}
	
	void validateDouble(double val) {
		//if (Double.isNaN(val) || Double.isInfinite(val)) {
		//	int n=1;
		//}
	}
}
