package com.physmo.neural.activations;

import com.physmo.neural.NodeLayer;

class None implements Activation {

	@Override
	public void CalculateActivation(NodeLayer nl) {
	}

	@Override
	public void CalculateDerivative(NodeLayer nl) {
		for (int i=0;i<nl.size;i++) {
			nl.derivatives[i]=1;//nl.values[i];
		}
	}
}
