// Copyright (c) 2018 Daniel Augusto Rizzi Salvadori. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

package tsne

import (
	"testing"
	"gonum.org/v1/gonum/mat"
)

// TestSquaredDistanceMatrix verifies that SquaredDistanceMatrix is behaving properly.
func TestSquaredDistanceMatrix(t *testing.T) {

	m := mat.NewDense(3, 2, []float64{2, 3, 8, 7, 2, 2})
	sd := SquaredDistanceMatrix(m)
	n, d := sd.Dims()
	if n != d {
		t.Error("SquaredDistanceMatrix is not square")
	}
	distMatData := mat.DenseCopyOf(sd).RawMatrix().Data
	expectedData := []float64{0, 52, 1, 52, 0, 61, 1, 61, 0}
	for i := 0; i < len(expectedData); i++ {
		if distMatData[i] != expectedData[i] {
			t.Error("SquaredDistanceMatrix failed")
		}
	}
}
