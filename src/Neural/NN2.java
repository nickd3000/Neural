package Neural;

import java.util.ArrayList;
import java.util.List;

// Rewriting neural network class with what I've learned...

public class NN2 {

	private static final String NEWLINE = "\n";
	List<NodeLayer> nodeLayers = new ArrayList<NodeLayer>();
	List<WeightLayer> weightLayers = new ArrayList<WeightLayer>();
	double learningRate = 0.1;
	double combinedError = 0;
	public NN2() {
		
	}
	
	public NN2 addLayer(int size) {
		size+=1; // Add bias node.
		
		NodeLayer lastLayer = getLastNodeLayer();
		int newLayerId = nodeLayers.size()+1;
		int requiredWeights = 0;
		
		if (lastLayer!=null) {
			requiredWeights = lastLayer.size*size;
		}

		NodeLayer newNodeLayer = new NodeLayer(size, newLayerId);
		nodeLayers.add(newNodeLayer);
		if (requiredWeights>0) weightLayers.add(new WeightLayer(requiredWeights, lastLayer, newNodeLayer));
		
		return this;
	}
	
	private NodeLayer getLastNodeLayer() {
		int numNodeLayers = nodeLayers.size();
		if (numNodeLayers<1) return null;
		return nodeLayers.get(numNodeLayers-1);
	}

	public NN2 randomizeWeights(double low, double high) {
		double span = high-low;
		for (WeightLayer wl : weightLayers) {
			for (int i=0;i<wl.weights.length;i++) {
				wl.weights[i]=low+(Math.random()*span);
			}
		}
		return this;
	}
	
	public NN2 learningRate(double value) {
		this.learningRate = value;
		return this;
	}
	
	public NN2 activationType(ActivationType actTp) {
		NodeLayer nl = getLastNodeLayer();
		nl.activationType = actTp;
		return this;
	}
	
	public void setInputValue(int i, double v) {
		nodeLayers.get(0).values[i]=v;
	}
	
	public void setOutputTargetValue(int i, double v) {
		getLastNodeLayer().targets[i]=v;
	}
	
	public double getOutputValue(int i) {
		return getLastNodeLayer().values[i];
	}
	
	public double getInnerValue(int layer, int node) {
		NodeLayer nl = nodeLayers.get(layer);
		return nl.values[node];
	}
	
	@Override
	public String toString() {
		String output = "NN2\n";
		
		for (NodeLayer nl : nodeLayers) {
			output += "NodeLayer " + nl.layerId + " size " + nl.size + NEWLINE;
		}
		for (WeightLayer wl : weightLayers) {
			output += "WeightLayer size" + wl.size + NEWLINE;
		}
		
		return output;
	}
	
	public void run(boolean learn) {
		
		for (WeightLayer wl : weightLayers) {
			
			resetBiasNode(wl.sourceNodeLayer);
			propogateLayerPair(wl);
			activateLayer(wl.targetNodeLayer);
		}
		
		for (NodeLayer nl : nodeLayers) {
			calculateLayerDerivatives(nl);
		}
		
		calculateLayerErrors(getLastNodeLayer());
		combinedError = sumLayerError(getLastNodeLayer());
		
		if (learn) {
			for (int i=weightLayers.size()-1;i>=0;i--) {
				backPropogateLayerPair(weightLayers.get(i));
			}
			
			for (WeightLayer wl : weightLayers) {	
				dampenDeltas(wl,0.99);
				updateDeltas(wl);
			}
			
			for (WeightLayer wl : weightLayers) {	
				adjustWeights(wl);
			}
		}
	}
	
	public double getCombinedError() {
		return combinedError;
	}
	
	private void propogateLayerPair(WeightLayer wl) {
		
		wl.targetNodeLayer.clearValues();
		
		double[] sourceValues = wl.sourceNodeLayer.values;
		double[] targetValues = wl.targetNodeLayer.values;
		double[] weights = wl.weights;
		
		int w=0;
		
		for (int sv = 0; sv<sourceValues.length;sv++) {
			for (int tv = 0; tv<targetValues.length;tv++) {
				targetValues[tv]+=sourceValues[sv]*weights[w++];
			}
		}
	}
	
	private void backPropogateLayerPair(WeightLayer wl) {

		double[] weights = wl.weights;		
		double[] sourceErrors = wl.sourceNodeLayer.errors;
		double[] sourceValues = wl.sourceNodeLayer.values;
		double[] sourceDerivatives = wl.sourceNodeLayer.derivatives;
		double[] targetErrors = wl.targetNodeLayer.errors;
		
		int w=0;
		wl.sourceNodeLayer.clearErrors();
		
		for (int sv = 0; sv<sourceErrors.length;sv++) {
			for (int tv = 0; tv<targetErrors.length;tv++) {
				//targetValues[tv]+=sourceValues[sv]*weights[w++];
				sourceErrors[sv]+=sourceDerivatives[sv]*weights[w++]*targetErrors[tv];
			}
		}
		
		// calc derivative for errors we just propogated back
//		for (int sv = 0; sv<sourceErrors.length;sv++) {
//			sourceErrors[sv] = derivativeTanh(sourceValues[sv])*sourceErrors[sv];
//		}
	}
	
	private void dampenDeltas(WeightLayer wl, double multiplier) {
		double[] deltas = wl.deltas;		

		for (int d = 0; d<deltas.length;d++) {
				
			deltas[d]*=multiplier;
			
		}
	}
	
	private void updateDeltas(WeightLayer wl) {
		double[] deltas = wl.deltas;		
		double[] sourceErrors = wl.sourceNodeLayer.errors;
		double[] targetErrors = wl.targetNodeLayer.errors;
		double[] sourceValues = wl.sourceNodeLayer.values;
		
		int w=0;
		
		for (int sv = 0; sv<sourceErrors.length;sv++) {
			for (int tv = 0; tv<targetErrors.length;tv++) {
				double delta = targetErrors[tv] * sourceValues[sv] * learningRate;
				deltas[w]+=delta;
				w++;
			}
		}
	}
	
	// Adjust weights using deltas.
	private void adjustWeights(WeightLayer wl) {
		double[] weights = wl.weights;	
		double[] deltas = wl.deltas;	
		double[] sourceErrors = wl.sourceNodeLayer.errors;
		double[] targetErrors = wl.targetNodeLayer.errors;
		double[] sourceValues = wl.sourceNodeLayer.values;
		
		int w=0;
		
		for (int sv = 0; sv<sourceErrors.length;sv++) {
			for (int tv = 0; tv<targetErrors.length;tv++) {
				//double delta = targetErrors[tv] * sourceValues[sv] * learningRate;
				weights[w]+=deltas[w];
				w++;
			}
		}
	}

		
	// TODO add different types of activation.
	private void activateLayer(NodeLayer nl) {
		if (nl.activationType==ActivationType.NONE) return;
		
		if (nl.activationType==ActivationType.TANH) {
			for (int i=0;i<nl.values.length;i++) {
				nl.values[i] = activationTanh(nl.values[i]);
			}
		}
		
		if (nl.activationType==ActivationType.RELU) {
			for (int i=0;i<nl.values.length;i++) {
				nl.values[i] = activationRelu(nl.values[i]);
			}
		}
		
		if (nl.activationType==ActivationType.LINEAR) {
			for (int i=0;i<nl.values.length;i++) {
				nl.values[i] = activationLinear(nl.values[i]);
			}
		}
		
		if (nl.activationType==ActivationType.SIGMOID) {
			for (int i=0;i<nl.values.length;i++) {
				nl.values[i] = activationSigmoid(nl.values[i]);
			}
		}
		
		if (nl.activationType==ActivationType.SOFTMAX) {
			activationSoftmax(nl);
		}
	}
	
	public void calculateLayerDerivatives(NodeLayer nl) {
		if (nl.activationType==ActivationType.NONE) {
			for (int i=0;i<nl.values.length;i++) {
				nl.derivatives[i] = nl.values[i];
			}
		}
		
		if (nl.activationType==ActivationType.TANH) {
			for (int i=0;i<nl.values.length;i++) {
				nl.derivatives[i] = derivativeTanh(nl.values[i]);
			}
		}
		
		if (nl.activationType==ActivationType.RELU) {
			for (int i=0;i<nl.values.length;i++) {
				nl.derivatives[i] = derivativeRelu(nl.values[i]);
			}
		}
		
		if (nl.activationType==ActivationType.LINEAR) {
			for (int i=0;i<nl.values.length;i++) {
				nl.derivatives[i] = derivativeLinear(nl.values[i]);
			}
		}
		
		if (nl.activationType==ActivationType.SIGMOID) {
			for (int i=0;i<nl.values.length;i++) {
				nl.derivatives[i] = derivativeSigmoid(nl.values[i]);
			}
		}
		
		if (nl.activationType==ActivationType.SOFTMAX) {
			for (int i=0;i<nl.values.length;i++) {
				nl.derivatives[i] = derivativeSoftmax(nl.values[i]);
			}
		}
	}
	
	// derivatives are required before this is executed.
	public void calculateLayerErrors(NodeLayer nl) {
		for (int i=0;i<nl.values.length;i++) {
			double e = nl.targets[i]-nl.values[i];
			nl.errors[i] = e * nl.derivatives[i];
		}
	}
	

	
	public double sumLayerError(NodeLayer nl) {
		double sum=0;
		for (int i=0;i<nl.values.length;i++) {
			sum+=Math.abs(nl.errors[i]);
		}
		return sum;
	}
	
	public void resetBiasNode(NodeLayer nl) {
		int numNodes = nl.size;
		nl.values[numNodes-1]=1.0f;
	}
	
	private double activationTanh(double x) {
		return Math.tanh(x);
	}
	
	private double derivativeTanh(double x) {
		return (1-(x*x));
	}
	
	private double activationRelu(double x) {
		return Math.max(0,x);
	}
	
	private double derivativeRelu(double x) {
		if (x<0) return 0.1;
		else return 1;
	}
	
	private double activationLinear(double x) {
		return x;
	}
	
	private double derivativeLinear(double x) {
		return 1;
	}
	
	private double activationSigmoid(double x) {
		return 1.0 / (1 + Math.exp(-1.0 * x));
	}
	
	private double derivativeSigmoid(double x) {
		return x * (1.0 - x);
	}
	
	private void activationSoftmax(NodeLayer nl) {

		double sum = 0;
		double max = -100;
		// add all outputs.
		for (int i=0;i<nl.size;i++) {
			double val = nl.values[i];
			val = Math.abs(val);
			sum += Math.exp(val);
			if (val>max) max=val;
		}
		
		// scale outputs by sum?
		for (int i=0;i<nl.size;i++) {
			double val = nl.values[i];
			nl.values[i]=(Math.exp(val))/sum;			
		}
	}
	
	private double derivativeSoftmax(double x) {
		return x * (1.0 - x);
	}
	
}
