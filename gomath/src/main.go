package main

import (
	"fmt"
	"log"
	"neuralnetwork"

	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/plotutil"
	"gonum.org/v1/plot/vg"
)

func main() {
	input := mat.NewDense(4, 3, []float64{
		0, 0, 1,
		0, 1, 1,
		1, 0, 1,
		1, 1, 1,
	})

	labels := mat.NewDense(4, 1, []float64{
		0,
		1,
		1,
		0,
	})
	numEpochs := 1500

	nn := neuralnetwork.NewSumSquaresNN(3, 4)
	lossHistory := make([]float64, numEpochs)

	var preds *mat.Dense
	for i := 0; i < numEpochs; i++ {
		preds = nn.Forward(input)
		nn.Backpropagation(input, labels, preds)
		lossHistory[i] = neuralnetwork.ComputeSumSquaresLoss(labels, preds)
	}

	fmt.Printf("predictions after %d epoch %v, expected %v\n", numEpochs, preds, labels)

	err := plotLoss(lossHistory)
	if err != nil {
		log.Fatalf("failed to plot: %v", err)
	}
}

func plotLoss(loss []float64) error {
	p, err := plot.New()
	if err != nil {
		return err
	}

	p.Title.Text = "Loss History"
	p.X.Label.Text = "Epochs"
	p.Y.Label.Text = "Loss"

	points := make(plotter.XYs, len(loss))
	for i := range loss {
		points[i].X = float64(i)
		points[i].Y = loss[i]
	}

	err = plotutil.AddLinePoints(p, "", points)
	if err != nil {
		return err
	}

	if err := p.Save(5*vg.Inch, 5*vg.Inch, "loss_history.png"); err != nil {
		return err
	}

	return nil
}
