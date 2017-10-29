package testNeuralNet;

import java.text.DecimalFormat;

import Neural.ActivationType;
import Neural.NN2;

public class TestNN2 {
	static DecimalFormat doubleFormat = new DecimalFormat("#.00"); 
	
	public static void main(String[] args) {
		basicTest();
	}

	public static void basicTest() {
		NN2 nn2 = new NN2()
				.addLayer(2)
				.activationType(ActivationType.NONE)
				.addLayer(2)
				.activationType(ActivationType.TANH)
				.addLayer(20)
				.activationType(ActivationType.TANH)
				.addLayer(2)
				.activationType(ActivationType.TANH)
				.randomizeWeights(-0.9, 0.9)
				.learningRate(0.1);
		
		nn2.setInputValue(0, 1);
		nn2.setInputValue(1, 0);
		nn2.setOutputTargetValue(0, 0);
		nn2.setOutputTargetValue(1, 1);
		
		System.out.println(nn2.toString());
		
		for (int i=0;i<200;i++) {
			if (Math.random()>0.5) {
				nn2.setInputValue(0, 1);
				nn2.setInputValue(1, 0);
				nn2.setOutputTargetValue(0, 0);
				nn2.setOutputTargetValue(1, 1);
			}
			else
			{
				nn2.setInputValue(0, 0);
				nn2.setInputValue(1, 1);
				nn2.setOutputTargetValue(0, 1);
				nn2.setOutputTargetValue(1, 0);
			}
			
			nn2.run(true);
			System.out.println("error: "+doubleFormat.format(nn2.getCombinedError()));
		}
		
		
	}
}
