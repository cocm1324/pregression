package pregression

import (
	"errors"
	"math"

	"gonum.org/v1/gonum/mat"
)

// Auto is to conduct polynomial regression and return optimal model based on AICc(Akaike Information Criterion)
// x is slice of input, y is slice of observation
// output is (slice of coefficient, R squared, error)
// slice of coefficient will have length of d + 1
// if length of x and y unmatches, it will return err
func Auto(x []float64, y []float64) ([]float64, float64, error) {
	if len(x) != len(y) {
		err := errors.New("polynomial-regression: length of x and y should be same")
		return nil, 0.0, err
	}

	var co []float64
	var rs float64
	var criterion float64 = math.MaxFloat64

	for d := 2; d < 10; d++ {
		_co, _ := FixedDegree(x, y, d)
		sse, sst, ssr := SseSstSsr(x, y, _co, d)
		_rs := sse / sst
		aic := AIC(len(y), d, ssr, true)

		if aic < criterion {
			criterion = aic
			rs = _rs
			co = _co
		}
	}

	return co, rs, nil
}

// FixedDegree is to conduct polynomial regression with fixed degree
// x is slice of input, y is slice of observation, d is degree of polynomial equation
// output is (slice of coefficient, error)
// slice of coefficient will have length of d + 1
// if length of x and y unmatches, it will return err
func FixedDegree(x []float64, y []float64, d int) ([]float64, error) {
	if len(x) != len(y) {
		err := errors.New("polynomial-regression: length of x and y should be same")
		return nil, err
	}

	bp := Vandermonde(x, d)
	yv := mat.NewVecDense(len(y), y)

	var solution mat.VecDense
	solution.SolveVec(bp, yv)

	coefficient := make([]float64, d+1)
	for i := 0; i < d+1; i++ {
		coefficient[i] = solution.AtVec(i)
	}

	return coefficient, nil
}

// AIC is function to calculate Akaike Information Criterion of model
// n is number of observation
// k is number of model parameter, in this case, degree of polynomial model
// rss is residual sum of square. this can be retreived by RssSst function
// correction is whether to apply correction for small number of observation
func AIC(_n, _k int, rss float64, correction bool) float64 {
	n := float64(_n)
	k := float64(_k)
	correctionTerm := float64(2*_k*(_k+1)) / float64(_n-_k-1)
	aic := n*math.Log(rss/n) + float64(2)*k + n*math.Log(float64(2)*math.Pi) + n
	if correction {
		return aic + correctionTerm
	}
	return aic
}

// BIC is function to calculate Bayesian Information Criterion of model
// n is number of observation
// k is number of model parameter, in this case, degree of polynomial model
// rss is residual sum of square. this can be retreived by RssSst function
func BIC(_n, _k int, rss float64) float64 {
	n := float64(_n)
	k := float64(_k)
	return n*math.Log(rss/n) + k*math.Log(n) + n*math.Log(float64(2)*math.Pi) + n
}

// RssSst is function to calculate Residual Sum of Squared (RSS) and Total Sum of Squared(SST)
// R squared can be calculated by dividing RSS by SST
func SseSstSsr(x []float64, y []float64, w []float64, d int) (float64, float64, float64) {
	mo := Calculate(x, w, d)
	var ym float64
	for _, v := range y {
		ym += v
	}
	ym = ym / float64(len(y))

	var sse, sst, ssr float64

	sse = 0.0
	sst = 0.0
	ssr = 0.0
	for i := 0; i < d+1; i++ {
		sse += math.Pow(mo[i]-ym, float64(2))
		sst += math.Pow(y[i]-ym, float64(2))
		ssr += math.Pow(mo[i]-y[i], float64(2))
	}

	return sse, sst, ssr
}

// Calculate is function to calculate model value of input
// x is input vector, w is coefficient of polynomial regression model, and d would be degree of the model.
func Calculate(x []float64, w []float64, d int) []float64 {
	v := Vandermonde(x, d)
	co := mat.NewDense(d+1, 1, w)

	var resultVec mat.Dense
	resultVec.Mul(v, co)

	result := make([]float64, d+1)
	for i := 0; i < d+1; i++ {
		result[i] = resultVec.At(i, 0)
	}
	return result
}

// Vandermonde is function to create vandermonde matrix.
// There is two parameters which are x and d
// x is inputs
// d is degree of Vandermonde matrix
// if d is 3, first row of matrix will be [1, x, x^2, x^3]
func Vandermonde(x []float64, d int) *mat.Dense {
	// making dataset
	var data []float64 = make([]float64, len(x)*(d+1))
	for i, v := range x {
		for j := 0; j < d+1; j++ {
			data[i*(d+1)+j] = math.Pow(v, float64(j))
		}
	}

	v := mat.NewDense(len(x), d+1, data)

	return v
}
