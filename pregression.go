package pregression

import (
	"encoding/json"
	"errors"
	"fmt"
	"math"

	"gonum.org/v1/gonum/mat"
)

const raw string = `[[46.22,324770300],[40.89,319110300],[46.52,314670300],[47.91,310030300],[45.24,304380300],[36.53,300310300],[46.73,291720300],[41.43,287400300],[38.22,279470300],[38.57,273060300],[41.99,269880300],[39.25,264230300],[42.96,258080300],[50.82,250790300],[49.37,243960300],[50.38,236600300],[62.46,227290300],[61.18,220670300],[62.76,212710300],[61.32,205910300],[54.12,199810300],[58.94,194880300],[48.15,189700300],[51.45,184530300],[67.71,177120300],[70.95,170740300],[48.5,164050300],[49.55,158440300],[44.76,153940300],[48.82,149540300],[37.89,145120300],[29.76,141240300],[24.91,138540300],[30.51,136070300],[25.34,133750300],[25.49,131220300],[24.84,126840300],[25.15,125944300],[31.9,125063300],[32.68,122433300],[34.08,117353300],[36.73,114533300],[49.88,104333300],[61.35,96523300],[57.9,88193300],[52.87,83973300],[51,77283300],[56.09,70823300],[44.58,66843300],[39.94,63063300],[34.3,60303300],[40.17,57193300],[36.53,53763300],[31.28,50943300],[30.11,47533300],[23.24,43903300],[20.54,40813300],[12.74,37493300],[10.24,35163300],[10.66,33303300],[9.18,31293300],[9.39,27613300],[5.76,25243300],[4.73,23893300],[4.87,22853300],[4.31,22137300],[4.52,21486300],[4.11,20386300],[4.11,19752300],[4.17,19632300],[3.8,18552300],[3.69,17957300],[3.47,17046300],[3.76,16563300],[3.76,16035300],[3.66,15756300],[3.41,15518300],[3.66,15251300],[3.61,14917300],[3.83,14549300],[3.59,14219300],[4.24,13615300],[3.48,13253300],[3.51,12937300],[3.51,12533300],[3.43,12127300],[3.48,11972300],[3.33,11483300],[3,11177300],[3.1,11002300],[3.06,10878300],[2.88,10758300],[2.95,10614300],[3.01,10492300],[2.87,10401800],[2.99,10257800],[2.87,10052800],[2.94,9378800],[2.8,8813800],[3.06,8289800],[2.9,7531800],[2.59,7094800],[2.61,6908800],[2.74,6691800],[2.45,6468800],[2.46,6206800],[2.06,5792800],[2.4,5193800],[2.64,4847800],[2.48,4475800],[2.66,4235800],[2.55,3841800],[2.25,3639800],[2.36,3473800],[2.35,3371800],[2.18,3170800],[2.17,3004800],[2.14,2929900],[2.16,2850000],[1.9,2780900],[1.74,2715200],[1.8,2618900],[1.45,2447900],[1.43,2251900],[1.38,2119900],[1.34,1990900],[1.41,1857900],[1.33,1674900],[1.36,1608800],[1.43,1504800],[1.3,1404900],[1.27,1140900],[1.29,1061100],[1.24,978900],[1.2,895300],[1.22,816400],[1.4,728900],[1.54,613900],[1.95,512900],[1.79,435000]]`

func main() {
	var d [][]float64
	err := json.Unmarshal([]byte(raw), &d)
	if err != nil {
		panic(err)
	}

	x := make([]float64, len(d))
	y := make([]float64, len(d))

	for i, v := range d {
		x[i] = v[1]
		y[i] = v[0]
	}

	co, rs, _ := Auto(x, y)

	fmt.Println(co, rs)
}

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
