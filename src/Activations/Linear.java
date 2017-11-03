package Activations;

import Neural.NodeLayer;

class Linear implements Activation {

	@Override
	public void CalculateActivation(NodeLayer nl) {
		for (int i=0;i<nl.size;i++) {
			nl.values[i] = nl.values[i];
		}
	}

	@Override
	public void CalculateDerivative(NodeLayer nl) {
		for (int i=0;i<nl.size;i++) {
			nl.derivatives[i]=1;
		}
	}
}
