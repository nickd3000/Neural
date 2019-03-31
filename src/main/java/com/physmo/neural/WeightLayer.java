package com.physmo.neural;

public class WeightLayer {
	int size;
	NodeLayer sourceNodeLayer = null;
	NodeLayer targetNodeLayer = null;
	
	double [] weights;
	double [] deltas; // Accumulated training deltas.
	
	public WeightLayer(int size, NodeLayer sourceLayer, NodeLayer targetLayer) {
		this.size=size;
		
		this.sourceNodeLayer = sourceLayer;
		this.targetNodeLayer = targetLayer;
		
		weights = new double[size];
		deltas = new double[size];
	}
}