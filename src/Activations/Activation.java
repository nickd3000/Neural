package Activations;

import Neural.NodeLayer;

public interface Activation {
	void CalculateActivation (NodeLayer nl);
	void CalculateDerivative (NodeLayer nl);
}