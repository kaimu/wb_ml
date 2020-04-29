package main

import (
	"flag"
	"fmt"
	"io/ioutil"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"time"
)

var python = flag.String("python", "python", "Python3 executable path")
var input = flag.String("input", "data", "input dataframes folder path")
var recInput = flag.String("rec_input", "", "recommendations input dataframes folder path")
var output = flag.String("output", "recommendations.csv", "output file path")

// Proxy args below to recommendation script
var trainingModel = flag.String("model", "als", "model to use for training: `als|nmslib_als|annoy_als|faiss_als|tfidf|cosine|bpr|lmf|bm25`")
var useGpu = flag.Bool("gpu", false, "use GPU (CUDA)")
var userLimit = flag.Uint("limit", 0, "limit the number or user to generate recommendations for")
var evaluate = flag.Bool("evaluate", false, "evaluate the model and print results before generating recommendations")

func main() {
	flag.Parse()
	// In case user provided invalid confidence strengths
	normalizeWeight()
	// Print params
	fmt.Printf("Python3 executable path (--python): '%v'\n", *python)
	fmt.Printf("Dataframes folder (--input): '%v'\n", *input)
	fmt.Printf("Recommendations will be written to (--output): '%v'\n", *output)
	fmt.Printf("Confidence strengths for actions: --view=%v --like=%v --favorite=%v\n\n", *weightView, *weightLike, *weightFavorite)
	// Create temp dir
	tmpDir, err := ioutil.TempDir(os.TempDir(), "wb_ml_")
	if err != nil {
		log.Fatalln(err)
	}
	defer os.RemoveAll(tmpDir)
	// Generate dataset
	datasetPath := filepath.Join(tmpDir, "dataset.tsv")
	_, err = generateDataset(*input, datasetPath)
	if err != nil {
		log.Println("Error:", err) // No `log.Fatalln` coz we want `defer` to clean temp-folder
		return
	}
	// Generate recommendations-dataset if required
	recDatasetPath := ""
	if *recInput != "" && *recInput != *input {
		recDatasetPath = filepath.Join(tmpDir, "rec_dataset.tsv")
		_, err = generateDataset(*recInput, recDatasetPath)
		if err != nil {
			log.Println("Error:", err)
			return
		}
	}
	// Run Collaborative Filtering recommendation script
	err = runCollaborativeFiltering(datasetPath, recDatasetPath, *output)
	if err != nil {
		log.Println("Error:", err)
		return
	}
	fmt.Printf("Finished")
}

func runCollaborativeFiltering(datasetPath, recDatasetPath, output string) (err error) {
	log.Println("Running Collaborative Filtering for the dataset...")
	start := time.Now()
	// Disable internal multithreading for OpenBLAS and Intel MKL
	os.Setenv("OPENBLAS_NUM_THREADS", "1")
	os.Setenv("MKL_NUM_THREADS", "1")
	// Run script with redirected outputs
	params := []string{
		"./recommend.py",
		"--input", datasetPath,
		"--rec_input", recDatasetPath,
		"--output", output,
		"--model", *trainingModel,
		"--limit", fmt.Sprint(*userLimit),
	}
	if *useGpu {
		params = append(params, "--gpu")
	}
	if *evaluate {
		params = append(params, "--evaluate")
	}
	cmd := exec.Command(*python, params...)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	err = cmd.Run()
	elapsed := time.Since(start)
	log.Printf("Collaborative Filtering results are generated to '%v' in %s", output, elapsed)
	return
}
