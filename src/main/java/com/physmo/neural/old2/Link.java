package com.physmo.neural.old2;

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
