package infer

// Prediction represents a class and it's associated score.
type Prediction struct {
	Class interface{}
	Score float32
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
	return ps[i].Score < ps[j].Score
}
