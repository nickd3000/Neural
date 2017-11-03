package Activations;

public enum ActivationType {
	NONE(new None()),
	TANH(new Tanh()),
	RELU(new Relu()),
	LINEAR(new Linear()),
	SIGMOID(new Sigmoid()), 
	SOFTMAX(new Softmax());
	
	private final Activation instance;
	
	ActivationType(Activation instance) {
		this.instance = instance;
	}
	
	public Activation getInstance() {
		return instance;
	}
	
}
