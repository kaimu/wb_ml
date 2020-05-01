package main

import (
	"compress/gzip"
	"encoding/csv"
	"flag"
	"fmt"
	"io"
	"io/ioutil"
	"log"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// CSVRecord contains a successfully parsed row of the CSV file
type CSVRecord struct {
	UserID  string
	Action  Action
	VideoID string
}

type Action = string
type VideosPerUser = map[string]uint8

const (
	actionView     Action = "success_view"
	actionLike     Action = "like"
	actionFavorite Action = "favorite"
)

var weightView = flag.Uint("view", 1, "'success_view' action weight `0-10`")
var weightLike = flag.Uint("like", 3, "'like' action weight `0-10`")
var weightFavorite = flag.Uint("favorite", 10, "'favorite' action weight `0-10`")

func normalizeWeight() {
	for _, v := range []*uint{weightView, weightLike, weightFavorite} {
		if *v > 10 {
			*v = 10
		}
	}
}

// generateDataset extracts all dataframes, unite them into a single dataset of format
// "<user_id>\t<video_id>\t<rating>"
func generateDataset(inputDir, datasetPath string) (users map[string]VideosPerUser, err error) {
	// List directory
	files, err := ioutil.ReadDir(inputDir)
	if err != nil {
		return
	}
	log.Printf("Processing files in '%v'...", inputDir)
	start := time.Now()
	// <user_id>:<filename>. Detects splitting a user between multiple dataframes
	var filePerUser = map[string]string{}
	// Set temporary dir and the dataset
	dataset, err := os.OpenFile(datasetPath, os.O_CREATE|os.O_RDWR, 0755)
	if err != nil {
		return
	}
	defer dataset.Close()
	// Process all the dataframes
	for _, f := range files {
		// Avoid non dataframe files
		if !strings.HasSuffix(f.Name(), ".gz") {
			continue
		}
		// A simple single level map with a key "<user_id>,<video_id>"  would be easier to work with,
		// however this way we avoid possible hash-collisions inside a large map
		users = map[string]VideosPerUser{}
		gz, e := os.Open(filepath.Join(inputDir, f.Name()))
		if e != nil {
			err = e
			return
		}
		defer gz.Close()
		// Extracting
		r, e := gzip.NewReader(gz)
		if e != nil {
			err = e
			return
		}
		defer r.Close()
		// Reading
		err = readDataframe(r, users, filePerUser, f.Name())
		if err != nil {
			return
		}
		// Writting to the dataset
		var row strings.Builder
		for userID, videos := range users {
			for videoID, weight := range videos {
				row.WriteString(fmt.Sprintf("%v\t%v\t%v\n", userID, videoID, weight))
				// Write to file 1MB blocks
				if row.Len() > 1048576 {
					_, err = dataset.WriteString(row.String())
					if err != nil {
						return
					}
					row.Reset()
				}
			}
		}
		if row.Len() > 0 {
			_, err = dataset.WriteString(row.String())
			if err != nil {
				return
			}
		}
		log.Printf("%v (users: %v)\n", f.Name(), len(users))
	}
	elapsed := time.Since(start)
	log.Printf("Dataset is created at '%v' in %s", datasetPath, elapsed)
	return
}

func readDataframe(r io.ReadCloser, users map[string]VideosPerUser, filePerUser map[string]string, filename string) (err error) {
	csvReader := csv.NewReader(r)
	csvReader.Comma = ' '
	for {
		// Read in a row. Check if we are at the end of the file
		record, e := csvReader.Read()
		if e == io.EOF {
			break
		} else if e != nil {
			return e
		}
		// Create a CSVRecord value for the row
		var r CSVRecord
		for colIndex, value := range record {
			if value == "" {
				err = fmt.Errorf("unexpected empty in column %d\n", colIndex)
				return
			}
			switch colIndex {
			case 0:
				r.UserID = value
			case 1:
				r.Action = value
			case 2:
				r.VideoID = value
			}
		}
		// We suggest that our dataframes are sorted by `user_id` and that a single `user_id` is contained in a single dataframe.
		// This allows us to avoid having a huge common map
		f, ok := filePerUser[r.UserID]
		if ok && f != filename {
			err = fmt.Errorf("same `user_id` exists in the separate dataframes")
		} else if !ok {
			filePerUser[r.UserID] = filename
		}
		if users[r.UserID] == nil {
			users[r.UserID] = VideosPerUser{}
		}
		// Update interaction weight for a video per user
		switch r.Action {
		case actionView:
			users[r.UserID][r.VideoID] = weight(users[r.UserID][r.VideoID], weightView)
		case actionLike:
			users[r.UserID][r.VideoID] = weight(users[r.UserID][r.VideoID], weightLike)
		case actionFavorite:
			users[r.UserID][r.VideoID] = weight(users[r.UserID][r.VideoID], weightFavorite)
		default:
			err = fmt.Errorf("unexpected action type '%v'\n", r.Action)
			return
		}
	}
	return
}

func weight(target uint8, source *uint) uint8 {
	// To avoid uint8 overflow
	if target <= 245 {
		return target + uint8(*source)
	}
	return target
}
