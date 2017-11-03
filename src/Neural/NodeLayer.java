package Neural;

import Activations.ActivationType;

public class NodeLayer {
	public int size;
	public int layerId;
	public ActivationType activationType;
	
	public double [] values;
	public double [] derivatives;
	public double [] targets;
	public double [] errors;
	
	public NodeLayer(int size, int layerId) {
		this.size=size;
		this.layerId=layerId;
		activationType = ActivationType.TANH;
		values = new double[size];
		derivatives = new double[size];
		targets = new double[size];
		errors = new double[size];
	}
	
	public void clearValues() {
		for (int i=0;i<values.length;i++) {
			values[i]=0;
			derivatives[i]=0;
			errors[i]=0;
			
		}
	}
	
	public void clearErrors() {
		for (int i=0;i<errors.length;i++) {
			errors[i]=0;
		}
	}

}