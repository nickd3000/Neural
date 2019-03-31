package com.physmo.neural.activations;

import com.physmo.neural.NodeLayer;

public interface Activation {
	void CalculateActivation (NodeLayer nl);
	void CalculateDerivative (NodeLayer nl);
}