package com.performance.cnn3d;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.CacheMode;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.Convolution3D;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.Subsampling3DLayer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;


/** @author Piotr Skoczek */

public final class NetworkUtils {

	private NetworkUtils() {

	}

	public static MultiLayerConfiguration createMultilayerConfiguration(int depth, int height, int width, int channels, int outputLabels)
	{
		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().seed( 123 )
				.weightInit( WeightInit.XAVIER_UNIFORM )
				.activation( Activation.RELU )
				.optimizationAlgo( OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT )
				.updater( new Adam( 0.001 ) )
				.biasUpdater( new Adam( 0.001 ) )
				.convolutionMode( ConvolutionMode.Same )
				.cacheMode( CacheMode.DEVICE )
				.miniBatch( true )
				.gradientNormalization( GradientNormalization.RenormalizeL2PerLayer )
				.l2( 0.001 )
				.list()//
				.layer( 0, new Convolution3D.Builder().kernelSize( 3, 3, 3 )
						.stride( 1, 1, 1 )
						.nOut( 30 )
						.weightInit( WeightInit.XAVIER_UNIFORM )
						.activation( Activation.RELU )
						.build() )//
				.layer( 1, new Subsampling3DLayer.Builder().kernelSize( 1, 2, 2 )
						.poolingType( Subsampling3DLayer.PoolingType.MAX )
						.convolutionMode( ConvolutionMode.Same )
						.build() )//
				.layer( 2, new Convolution3D.Builder().kernelSize( 3, 3, 3 )
						.stride( 1, 1, 1 )
						.weightInit( WeightInit.XAVIER_UNIFORM )
						.activation( Activation.RELU )
						.nOut( 40 )
						.build() )//
				.layer( 3, new Subsampling3DLayer.Builder().kernelSize( 2, 2, 2 )
						.poolingType( Subsampling3DLayer.PoolingType.MAX )
						.convolutionMode( ConvolutionMode.Same )
						.build() )//
				.layer( 4, new Convolution3D.Builder().kernelSize( 3, 3, 3 )
						.stride( 2, 2, 2 )
						.weightInit( WeightInit.XAVIER_UNIFORM )
						.activation( Activation.RELU )
						.nOut( 50 )
						.build() )//
				.layer( 5, new Subsampling3DLayer.Builder().kernelSize( 2, 2, 2 )
						.poolingType( Subsampling3DLayer.PoolingType.MAX )
						.convolutionMode( ConvolutionMode.Same )
						.build() )//
				.layer( 6, new DenseLayer.Builder().nOut( 250 ).dropOut( 0.7 ).build() )
				.layer( 7, new OutputLayer.Builder( LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD ).activation( Activation.SOFTMAX )
						.nOut( outputLabels )
						.build() )//
				.setInputType( InputType.convolutional3D( depth, height, width, channels ) )

				.build();

		return conf;
	}

}
