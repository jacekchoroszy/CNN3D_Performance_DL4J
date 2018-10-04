package com.performance.cnn3d;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.PerformanceListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.dataset.DataSet;

/** @author Piotr Skoczek */

public class Runner {

	private static final int HEIGHT = 96, WIDTH = 96, CHANNELS = 1, DEPTH = 16, OUTPUT_LABELS = 2, BATCH_SIZE = 32;

	private static int TOTAL_ITERATIONS = 35;

	public static void main(String[] args) {
		RandomDataSetProvider randomProvider = new RandomDataSetProvider(HEIGHT, WIDTH, CHANNELS, DEPTH, BATCH_SIZE,
				OUTPUT_LABELS);

		MultiLayerConfiguration configuration = NetworkUtils.createMultilayerConfiguration(DEPTH, HEIGHT, WIDTH,
				CHANNELS, OUTPUT_LABELS);

		//		Nd4j.getMemoryManager().togglePeriodicGc(false);
		// Nd4j.getMemoryManager().setAutoGcWindow(5000);

		MultiLayerNetwork network = new MultiLayerNetwork(configuration);

		network.setListeners(new ScoreIterationListener(1), new PerformanceListener(1));
		network.init();

		int idx=0;
		while (true) {
			// features shape: [2, 1, 16, 96, 96] [minibatchsize, channels, depth, height, width]
			// labels shape: [2, 2] [minibatchsize, vector array size]
			DataSet dataSet = randomProvider.generateNextDataSet();

			network.fit(dataSet);

			idx++;
			if( idx == TOTAL_ITERATIONS ) {
				System.exit(0);
			}
		}

	}

}
