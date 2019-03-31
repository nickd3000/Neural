package com.physmo.neural.activations;

import com.physmo.neural.NodeLayer;

class Tanh implements Activation {

	@Override
	public void CalculateActivation(NodeLayer nl) {
		for (int i=0;i<nl.size;i++) {
			nl.values[i] = Math.tanh(nl.values[i]);
		}
	}

	@Override
	public void CalculateDerivative(NodeLayer nl) {
		for (int i=0;i<nl.size;i++) {
			nl.derivatives[i]=(1-(nl.values[i]*nl.values[i]));
		}
	}
}
