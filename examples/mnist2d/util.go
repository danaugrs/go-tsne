package main

import (
	"bufio"
	"encoding/csv"
	"io"
	"log"
	"os"
	"strconv"

	"gonum.org/v1/gonum/mat"
)

func LoadMNIST() (mat.Matrix, mat.Matrix) {

	return CSVmatrix("../data/mnist2500_X.csv"), CSVmatrix("../data/mnist2500_labels.csv")
}

func CSVmatrix(filepath string) mat.Matrix {

	csvFile, err := os.Open(filepath)
	if err != nil {
		log.Fatal(err)
	}
	reader := csv.NewReader(bufio.NewReader(csvFile))
	var data []float64
	var r, c int
	firstLine := true
	for {
		line, err := reader.Read()
		if err == io.EOF {
			break
		} else if err != nil {
			log.Fatal(err)
		}
		if firstLine {
			c = len(line)
		}
		for i := range line {
			f, err := strconv.ParseFloat(line[i], 64)
			if err != nil {
				log.Fatal(err)
			}
			data = append(data, f)
		}
		r++
	}

	return mat.NewDense(r, c, data)
}
