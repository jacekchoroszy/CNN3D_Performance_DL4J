package com.performance.cnn3d;

import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.optimize.api.BaseTrainingListener;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class TimeScoreIterationListener extends BaseTrainingListener {
	private static final Logger log = LoggerFactory.getLogger(TimeScoreIterationListener.class);

	private long lastTimeMillis;

	@Override
	public void iterationDone(Model model, int iteration, int epoch) {
		double score = model.score();

		if (lastTimeMillis == 0) {
			log.info("Score at iteration {} is {}", iteration, score);
			lastTimeMillis = System.currentTimeMillis();
		} else {
			long currentTimeMillis = System.currentTimeMillis();
			log.info("Score at iteration {} is {}, iteration time is {} ms", iteration, score,
					(currentTimeMillis - lastTimeMillis));
			lastTimeMillis = currentTimeMillis;
		}
	}

}
