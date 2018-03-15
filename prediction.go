package infer

// Prediction represents a value and it's associated
// confidence score.
type Prediction struct {
	Value      interface{} `json:"value"`
	Confidence float32     `json:"confidence"`
}

// Predictions is a list of Prediction.
type Predictions []*Prediction

func (ps Predictions) Len() int {
	return len(ps)
}

func (ps Predictions) Swap(i, j int) {
	ps[i], ps[j] = ps[j], ps[i]
}

func (ps Predictions) Less(i, j int) bool {
	return ps[i].Confidence < ps[j].Confidence
}
