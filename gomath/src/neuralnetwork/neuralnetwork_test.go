package neuralnetwork

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"gonum.org/v1/gonum/mat"
)

// Note: All the tests desired output has been generated with numpy.
func TestForward(t *testing.T) {
	nn := SumSquaresNN{
		w1: mat.NewDense(2, 2, []float64{
			1, 0,
			0, 1,
		}),
		w2:        mat.NewDense(2, 1, []float64{1, 2}),
		nHidden:   2,
		inputSize: 2,
	}

	input := mat.NewDense(2, 2, []float64{
		2, 1,
		1, 3,
	})

	actual := nn.Forward(input)
	assertDenseAlmostEqual(t, mat.NewDense(2, 1, []float64{0.91236936, 0.93315575}), actual)
}

func TestComputeSquareLoss(t *testing.T) {
	testcases := []struct {
		name     string
		v1       *mat.Dense
		v2       *mat.Dense
		expected float64
	}{
		{
			name: "equal",
			v1: mat.NewDense(2, 2, []float64{
				2,
				1,
				3,
				1,
			}),
			v2: mat.NewDense(2, 2, []float64{
				2,
				1,
				3,
				1,
			}),
			expected: 0,
		},
		{
			name: "different",
			v1: mat.NewDense(2, 2, []float64{
				2,
				1,
				3,
				1,
			}),
			v2: mat.NewDense(2, 2, []float64{
				1,
				3,
				3,
				1,
			}),
			expected: 5,
		},
	}

	for _, tt := range testcases {
		t.Run(tt.name, func(t *testing.T) {
			actual := ComputeSumSquaresLoss(tt.v1, tt.v2)
			assert.InDelta(t, tt.expected, actual, 1e-4)
		})
	}
}

func TestBackpropagation(t *testing.T) {
	nn := SumSquaresNN{
		w1: mat.NewDense(2, 2, []float64{
			1, 3,
			4, 1,
		}),
		w2: mat.NewDense(2, 1, []float64{
			3,
			2,
		}),
		nHidden:   2,
		inputSize: 2,
	}

	input := mat.NewDense(2, 2, []float64{
		2, 1,
		1, 3,
	})

	preds := mat.NewDense(2, 1, []float64{
		0.1,
		0.9,
	})

	labels := mat.NewDense(2, 1, []float64{
		0,
		1,
	})

	nn.Forward(input)
	nn.Backpropagation(input, labels, preds)
	desiredW1 := mat.NewDense(2, 2, []float64{
		0.9997337390537693, 3.00002325840952,
		3.999867174669975, 1.0002336150409825,
	})
	desiredW2 := mat.NewDense(2, 1, []float64{
		3.00002810829532,
		1.999955533469018,
	})

	assertDenseAlmostEqual(t, desiredW1, nn.w1)
	assertDenseAlmostEqual(t, desiredW2, nn.w2)
}

func assertDenseAlmostEqual(t *testing.T, desired, actual *mat.Dense) {
	desiredCols, desiredRows := desired.Dims()
	actualCols, actualRows := actual.Dims()

	assert.Equal(t, desiredCols, actualCols)
	assert.Equal(t, desiredRows, actualRows)

	var diff mat.Dense
	diff.Sub(desired, actual)
	norm := mat.Norm(&diff, 2)
	assert.InDelta(t, 0, norm, 1e-4)
}
