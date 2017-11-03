package Activations;

import Neural.NodeLayer;

class Softmax implements Activation {

	@Override
	public void CalculateActivation(NodeLayer nl) {

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

	@Override
	public void CalculateDerivative(NodeLayer nl) {
		for (int i=0;i<nl.size;i++) {
			nl.derivatives[i]=nl.values[i] * (1.0 - nl.values[i]);
		}
	}
}
