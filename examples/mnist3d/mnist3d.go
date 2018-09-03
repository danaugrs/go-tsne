// Copyright (c) 2018 Daniel Augusto Rizzi Salvadori. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

package main

import (
	"github.com/danaugrs/go-tsne/tsne"
	"github.com/danaugrs/go-tsne/examples/data"
	"github.com/sjwhitworth/golearn/pca"
	"gonum.org/v1/gonum/mat"
	"github.com/g3n/engine/util/application"
	"github.com/g3n/engine/math32"
	"github.com/g3n/engine/material"
	"github.com/g3n/engine/graphic"
	"github.com/g3n/engine/light"
	"github.com/g3n/engine/texture"
	"github.com/g3n/engine/window"
	"github.com/g3n/engine/core"
	"github.com/g3n/engine/gui"
	"time"
	"math/rand"
	"image"
	"image/color"
)

const Credits string = "Open source software by Daniel Salvadori (github.com/danaugrs/tsne). Written in Go and powered by g3n (github.com/g3n/engine)."
const Instructions string = "(PageUp/PageDown) Change sprite size\n(F) Toggle Fullscreen\n(R) Toggle Rotation\n(ESC) Quit"

// ExampleMNIST3D is this example application displaying t-SNE being performed in real-time, in 3D,
// using a subset of the MNIST dataset of handwritten digits.
type ExampleMNIST3D struct {
	app         *application.Application
	sprites     []*graphic.Sprite
	spriteScene *core.Node
	spriteScale float32
	rotating    bool
}

func main() {

	// Create application
	ex := new(ExampleMNIST3D)
	ex.app, _ = application.Create(application.Options{
		Title:  "t-SNE MNIST 3D",
		Width:  1024,
		Height: 768,
	})

	// Disable zoom
	ex.app.Orbit().ZoomSpeed = 0

	// Change background to white and reposition camera
	ex.app.Gl().ClearColor(0.9,0.9,0.9,1)
	ex.app.CameraPersp().SetPosition(0, 0, 12)

	// Add lights to the scene
	ambientLight := light.NewAmbient(&math32.Color{1.0, 1.0, 1.0}, 0.8)
	ex.app.Scene().Add(ambientLight)
	pointLight := light.NewPoint(&math32.Color{1, 1, 1}, 5.0)
	pointLight.SetPosition(1, 0, 2)
	ex.app.Scene().Add(pointLight)

	// Add credits
	creditsLabel := gui.NewImageLabel(Credits)
	creditsLabel.SetColor(&math32.Color{0,0,0})
	ex.app.Gui().Subscribe(gui.OnResize, func(evname string, ev interface{}) {
		creditsLabel.SetWidth(ex.app.Gui().ContentWidth())
		creditsLabel.SetPositionY(ex.app.Gui().ContentHeight() - 2*creditsLabel.ContentHeight())
	})
	ex.app.Gui().Add(creditsLabel)

	// Add instructions
	instructionsLabel := gui.NewImageLabel(Instructions)
	instructionsLabel.SetColor(&math32.Color{0,0,0})
	ex.app.Gui().Subscribe(gui.OnResize, func(evname string, ev interface{}) {
		instructionsLabel.SetPositionX(10)
		instructionsLabel.SetPositionY(10)
	})
	ex.app.Gui().Add(instructionsLabel)

	// Add loading text
	loadingLabel := gui.NewImageLabel("Computing P-values...")
	loadingLabel.SetFontSize(32)
	loadingLabel.SetColor(&math32.Color{0,0,0})
	ex.app.Gui().Subscribe(gui.OnResize, func(evname string, ev interface{}) {
		loadingLabel.SetPositionX((ex.app.Gui().ContentWidth() - loadingLabel.ContentWidth())/2)
		loadingLabel.SetPositionY((ex.app.Gui().ContentHeight() - loadingLabel.ContentHeight())/2)
	})
	ex.app.Gui().Add(loadingLabel)

	// Dispatch a fake OnResize event to update all subscribed elements
	ex.app.Gui().Dispatch(gui.OnResize, nil)

	// Initialize the random seed
	rand.Seed(int64(time.Now().Nanosecond()))

	// Load a subset of MNIST with 2500 records
	X, Y := data.LoadMNIST()

	// Create the digit sprites
	ex.createDigitSprites(X, Y)

	// Subscribe to window key events
	ex.app.Window().Subscribe(window.OnKeyDown, func(evname string, ev interface{}) {
		kev := ev.(*window.KeyEvent)
		// ESC terminates program
		if kev.Keycode == window.KeyEscape {
			ex.app.Quit()
			return
		}
		// F toggles full screen
		if kev.Keycode == window.KeyF {
			ex.app.Window().SetFullScreen(!ex.app.Window().FullScreen())
			return
		}
		// R toggles rotation
		if kev.Keycode == window.KeyR{
			ex.rotating = !ex.rotating
			return
		}
	})

	// Subscribe to before render events to check for KeyState
	ex.app.Subscribe(application.OnBeforeRender, func(evname string, ev interface{}) {
		// TODO Interpolate embedding positions?
		if ex.app.KeyState().Pressed(window.KeyPageUp) {
			ex.spriteScale += 0.02
			ex.scaleSprites()
		}
		if ex.app.KeyState().Pressed(window.KeyPageDown) {
			ex.spriteScale -= 0.02
			ex.scaleSprites()
		}
		if ex.rotating {
			ex.spriteScene.RotateY(0.01)
		}
	})

	// Pre-process the data with PCA (Principal Component Analysis)
	// reducing the number of dimensions from 784 (28x28) to the top 100 principal components
	Xdense := mat.DenseCopyOf(X)
	pcaTransform := pca.NewPCA(100)
	Xt := pcaTransform.FitTransform(Xdense)

	// Create the t-SNE dimensionality reductor and embed the MNIST data in 3D
	t := tsne.NewTSNE(3, 500, 500, 300,true)
	go t.EmbedData(Xt, func(iter int, divergence float64, embedding mat.Matrix) bool {
		if iter == 0 {
			loadingLabel.SetVisible(false)
			ex.app.Scene().Add(ex.spriteScene)
			ex.rotating = true
		}
		ex.updateSprites(embedding)
		return false
	})

	// Start the 3D application
	ex.app.Run()
}

// scaleSprites scales all the sprites to match spriteScale.
func (ex *ExampleMNIST3D) scaleSprites() {

	for i := 0; i < len(ex.sprites); i++ {
		ex.sprites[i].SetScale(ex.spriteScale, ex.spriteScale, ex.spriteScale)
	}
}

// updateSprites updates the position of the sprites based on the specified embedding matrix.
func (ex *ExampleMNIST3D) updateSprites(Y mat.Matrix) {

	const distScale = float32(6)
	// Update positions and calculate maxDist
	n, _ := Y.Dims()
	posVec := math32.NewVec3()
	var maxDist float32
	for i := 0; i < n; i++ {
		posVec.Set(float32(Y.At(i, 0)), float32(Y.At(i, 1)), float32(Y.At(i, 2)))
		ex.sprites[i].SetPositionVec(posVec)
		dist := posVec.Length()
		if dist > maxDist {
			maxDist = dist
		}
	}
	// Normalize positions
	for i := 0; i < n; i++ {
		pos := ex.sprites[i].Position()
		pos.MultiplyScalar(distScale /maxDist)
		ex.sprites[i].SetPositionVec(&pos)
	}
}

// createDigitSprites creates a sprite for each digit in the dataset.
func (ex *ExampleMNIST3D) createDigitSprites(X mat.Matrix, Y mat.Matrix) {

	// Create scene to contain sprites
	ex.spriteScene = core.NewNode()
	// Add an axis helper to the scene
	ex.spriteScene.Add(graphic.NewAxisHelper(6))
	// Define colors for the digit classes
	classColor := []color.RGBA{
		{255,0,0, 0xff},
		{255,255,0, 0xff},
		{0,255,0, 0xff},
		{0,255,255, 0xff},
		{0,0,255, 0xff},
		{255,0,255, 0xff},
		{0,0,0, 0xff},
		{128,128,128, 0xff},
		{255,128,0, 0xff},
		{128,64,0, 0xff},
	}
	// Generate a sprite for each digit in the dataset and color each based on the digit's class
	ex.sprites = make([]*graphic.Sprite, 0)
	ex.spriteScale = 1
	const imgSide = 28
	n, _ := X.Dims()
	Xdense := mat.DenseCopyOf(X)
	for i := 0; i < n; i++ {
		img := image.NewRGBA(image.Rectangle{image.Point{0, 0}, image.Point{imgSide, imgSide}})
		Digit := Xdense.RowView(i)
		// Set color for each pixel.
		for x := 0; x < imgSide; x++ {
			for y := 0; y < imgSide; y++ {
				if Digit.AtVec(x*imgSide+ y) == 1 {
					img.Set(x, y, color.Transparent)
				} else {
					img.Set(x, y, classColor[int(Y.At(i, 0))])
				}
			}
		}
		// Create sprite material with digit texture
		spriteMat := material.NewStandard(&math32.Color{1, 1, 1})
		spriteMat.AddTexture(texture.NewTexture2DFromRGBA(img))
		spriteMat.SetTransparent(true)
		// Create sprite and add to list
		spriteMesh := graphic.NewSprite(0.3, 0.3, spriteMat)
		ex.spriteScene.Add(spriteMesh)
		ex.sprites = append(ex.sprites, spriteMesh)
	}
}
