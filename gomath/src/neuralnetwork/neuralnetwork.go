package neuralnetwork

import (
	"math"
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

// NeuralNetwork represets a Neural Network
type NeuralNetwork interface {
	// Forward computes forward propagation.
	Forward(*mat.Dense) *mat.Dense
	// Backward computes backward propagation.
	Backward(*mat.Dense, *mat.Dense, *mat.Dense)
}

// SumSquaresNN represents a neural network with two layers.
// The activation is a sigmoid function.
// The loss is the sum of the squares sum(pred - y)^2.
// The output has size 1.
type SumSquaresNN struct {
	w1         *mat.Dense
	w2         *mat.Dense
	firstLayer *mat.Dense
	nHidden    int
	inputSize  int
}

// NewSumSquaresNN returns a new SumSquaresNN.
func NewSumSquaresNN(inputSize, hiddenNeurons int) *SumSquaresNN {
	data := make([]float64, inputSize*hiddenNeurons)
	for i := range data {
		data[i] = rand.NormFloat64()
	}
	w1 := mat.NewDense(inputSize, hiddenNeurons, data)

	data = make([]float64, hiddenNeurons)
	for i := range data {
		data[i] = rand.NormFloat64()
	}
	w2 := mat.NewDense(hiddenNeurons, 1, data)

	return &SumSquaresNN{
		w1:        w1,
		w2:        w2,
		nHidden:   hiddenNeurons,
		inputSize: inputSize,
	}
}

func ComputeSumSquaresLoss(labels, preds *mat.Dense) float64 {
	var diff mat.Dense
	diff.Sub(preds, labels)

	r, c := diff.Dims()
	pow := mat.NewDense(r, c, nil)

	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			pow.Set(i, j, diff.At(i, j)*diff.At(i, j))
		}
	}

	return mat.Sum(pow)
}

// Forward computes the forward propagation for a SumSquaresNN.
func (n *SumSquaresNN) Forward(x *mat.Dense) *mat.Dense {
	var layerOne mat.Dense
	layerOne.Mul(x, n.w1)
	layerOne = sigmoidMatrix(layerOne)
	n.firstLayer = &layerOne

	var layerTwo mat.Dense
	layerTwo.Mul(&layerOne, n.w2)
	layerTwo = sigmoidMatrix(layerTwo)

	return &layerTwo
}

// Backpropagation computes one step of backpropagation for the network.
func (n *SumSquaresNN) Backpropagation(x, labels, preds *mat.Dense) {
	// Using the chain rule (where  s = sigmoid, z1 = xW1, z2 = s(xW1)W2, L = sum(s(z2) - y)^2):
	// dL/dW2 = dL/ds(z2) * ds(z2)/dz2 * dz2/dW2
	// dL/dW1 = dL/ds(z2) * ds(z2)/dz2 * dz2/ds(z1)* ds(z1)/dz1 * dz1/dW1
	bs, _ := preds.Dims()

	dldsz2 := mat.NewDense(bs, 1, nil)
	dldsz2.Sub(preds, labels)
	dldsz2.Scale(2, dldsz2)

	dsz2dz2 := derivativeFromSigmoidMatrix(*preds)
	var dLdW2 mat.Dense
	dLdW2.MulElem(dldsz2, &dsz2dz2)
	dLdW2.Mul(n.firstLayer.T(), &dLdW2)

	dldsz2.MulElem(dldsz2, &dsz2dz2)
	partial := mat.NewDense(bs, n.nHidden, nil)
	partial.Mul(dldsz2, n.w2.T())

	dsz1dz1 := derivativeFromSigmoidMatrix(*n.firstLayer)
	partial.MulElem(partial, &dsz1dz1)

	dLdW1 := mat.NewDense(n.inputSize, n.nHidden, nil)
	dLdW1.Mul(x.T(), partial)

	n.w1.Sub(n.w1, dLdW1)
	n.w2.Sub(n.w2, &dLdW2)
}

func sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

func sigmoidMatrix(x mat.Dense) mat.Dense {
	r, c := x.Dims()
	sx := mat.NewDense(r, c, nil)

	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			sx.Set(i, j, sigmoid(x.At(i, j)))
		}
	}

	return *sx
}

// Note: this function computes the derivative given the sigmoid.
func derivativeFromSigmoidMatrix(x mat.Dense) mat.Dense {
	r, c := x.Dims()
	sx := mat.NewDense(r, c, nil)

	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			v := x.At(i, j)
			sx.Set(i, j, v*(1-v))
		}
	}

	return *sx
}
