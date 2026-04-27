#pragma once
#include <cstddef>
#include <memory>
#include <type_traits>
#include <functional>
#include <cassert>
#include <random>

// hardware acceleration 
#include <Accelerate/Accelerate.h>

template <typename T>
concept Numeric = std::is_floating_point_v<T>;

template <Numeric T>
class Matrix{
private:
	std::unique_ptr<T[]> data;	// unique ptr to create one continous data area for easy chache 

public:
	size_t rows, cols;
	
	// constructors 
	Matrix();
	Matrix(size_t rows, size_t cols);
	Matrix(size_t rows, size_t cols, const T* src);

	// rule of 5
	Matrix(const Matrix& other); // copy matrix(matrix other)
	Matrix(Matrix&& other) noexcept;
	Matrix& operator=(const Matrix& other); // copy with =
	Matrix& operator=(Matrix&& other) noexcept;
	~Matrix() = default;

	// element access 
	T& operator()(size_t i, size_t j);
	const T& operator()(size_t i, size_t j) const;

	// matrix arithematic 
	Matrix<T> operator+(const Matrix<T>& other) const;
	Matrix<T> operator-(const Matrix<T>& other) const;
	Matrix<T> operator*(T scaler) const;
	Matrix<T> operator*(const Matrix<T>& other) const;  // elementwise
	Matrix<T> matmul(const Matrix<T>& other) const;

	// transpose 
	Matrix<T> transpose() const;

	// apply 
	template <typename F>
	Matrix<T> apply(F f) const;

	template <typename F>
	void apply_inplace(F f);

	// factories 
	static Matrix<T> zeros(size_t rows, size_t cols);
	static Matrix<T> random(size_t rows, size_t cols, T low=0.0, T high=1.0);
	static Matrix<T> he(size_t rows, size_t cols);
};

// matrix implimentation --------------------------------------------------------------------------------------------

// constructors
// default constructor 

template<Numeric T>
Matrix<T>::Matrix() : rows(0), cols(0), data(nullptr) {}

template<Numeric T>	
Matrix<T>::Matrix(size_t rows, size_t cols)
	:rows(rows), cols(cols), data(std::make_unique<T[]>(rows*cols))
{}

// constructor with data
template<Numeric T>
Matrix<T>::Matrix(size_t rows, size_t cols, const T* src)
    : rows(rows), cols(cols), data(rows*cols ? std::make_unique<T[]>(rows*cols) : nullptr) {
        if (src && rows*cols) std::copy(src, src + rows*cols, data.get());
}

// copy constructor
template<Numeric T>
Matrix<T>::Matrix(const Matrix& other)
	:Matrix(other.rows, other.cols, other.data.get())
{}

// move operator
template<Numeric T>
Matrix<T>::Matrix(Matrix&& other) noexcept
	:rows(other.rows), cols(other.cols), data(std::move(other.data)) {
	other.rows = 0; other.cols = 0;
}

// copy assignment 
template<Numeric T>
Matrix<T>& Matrix<T>::operator=(const Matrix& other) {
	if(this == &other) return *this;

	Matrix<T> temp(other);
	std::swap(rows, temp.rows);
	std::swap(cols, temp.cols);
	std::swap(data, temp.data);

	return *this;
}

// move assignment 
template<Numeric T>
Matrix<T>& Matrix<T>::operator=(Matrix&& other) noexcept {
	if(this == &other) return *this;
	rows=other.rows; cols=other.cols;
	data = std::move(other.data);
	other.rows=0; other.cols=0; 
	return *this;
}

// element Access
template<Numeric T>
T& Matrix<T>::operator()(size_t i, size_t j) {
	assert(i<rows && j<cols);
	return data[i * cols + j];
}

template<Numeric T>
const T& Matrix<T>::operator()(size_t i, size_t j) const {
	assert(i<rows && j<cols);
	return data[i * cols + j];
}

// matrix arithematic
// matrix addition
template<Numeric T>
Matrix<T> Matrix<T>::operator+(const Matrix<T>& other) const {
	assert(this->rows == other.rows && this->cols == other.cols);

	Matrix<T> res(rows, cols);
	for(size_t i=0; i<rows; i++){
		for(size_t j=0; j<cols; j++){
			res(i,j) = (*this)(i,j) + other(i,j);
		}
	}
	return res;
}

// matrix subtraction
template <Numeric T>
Matrix<T> Matrix<T>::operator-(const Matrix<T>& other) const{
	assert(this->rows == other.rows && this->cols == other.cols);

	Matrix<T> res(rows, cols);
	for(size_t i=0; i<rows; i++){
		for(size_t j=0; j<cols; j++){
			res(i,j) = (*this)(i,j) - other(i,j);
		}
	}
	return res;
}

// scaler multiplication
template<Numeric T>
Matrix<T> Matrix<T>::operator*(T scaler) const {
	Matrix<T> res(rows, cols);
	for(size_t i=0; i<rows; i++){
		for(size_t j=0; j<cols; j++){
			res(i,j) = (*this)(i,j) * scaler;
		}
	}
	return res;
}

// elementwise multiplication
template<Numeric T>
Matrix<T> Matrix<T>::operator*(const Matrix<T>& other) const {
    assert(rows == other.rows && cols == other.cols);
    Matrix<T> res(rows, cols);
    for(size_t i=0; i<rows; i++)
        for(size_t j=0; j<cols; j++)
            res(i,j) = (*this)(i,j) * other(i,j);
    return res;
}

// matrix multiplication
template<Numeric T>
Matrix<T> Matrix<T>::matmul(const Matrix<T>& other) const {
    assert(this->cols == other.rows);
    Matrix<T> res(this->rows, other.cols);

    if constexpr (std::is_same_v<T, float>) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    this->rows, other.cols, this->cols,
                    1.0f,
                    data.get(), this->cols,
                    other.data.get(), other.cols,
                    0.0f,
                    res.data.get(), other.cols);
    } else {
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    this->rows, other.cols, this->cols,
                    1.0,
                    data.get(), this->cols,
                    other.data.get(), other.cols,
                    0.0,
                    res.data.get(), other.cols);
    }

    return res;
}

// COMMENTED FOR BETTER APPROACH, NOT DELETED TO KEEP ORIGINAL APPROACH FOR TESTING
// template<Numeric T>
// Matrix<T> Matrix<T>::matmul(const Matrix<T>& other) const{
// 	assert(this->cols == other.rows);
// 	Matrix<T> res(this->rows, other.cols);
//
// 	for(size_t i=0; i<this->rows;i++){
// 		for(size_t j=0; j<other.cols; j++){
// 			for(size_t k=0; k<this->cols; k++){
// 				res(i,j) += (*this)(i,k) * other(k,j);	// dependent on res being 0 initilised | can break
// 			}
// 		}
// 	}
// 	return res;
// }

// transpose 
template<Numeric T>
Matrix<T> Matrix<T>::transpose() const {
    Matrix<T> res(cols, rows);
    for (size_t i = 0; i < rows; i++)
        for (size_t j = 0; j < cols; j++)
            res(j, i) = (*this)(i, j);
    return res;
}

// apply functions 
template<Numeric T>
template <typename F>
Matrix<T> Matrix<T>::apply(F f) const{
	Matrix<T> res(rows, cols);

	for(size_t i=0; i<rows; i++){
		for(size_t j=0; j<cols; j++){
			res(i,j) = f((*this)(i,j));
		}
	}
	return res;
}

template<Numeric T>
template <typename F>
void Matrix<T>::apply_inplace(F f){
	for(size_t i=0; i<rows; i++){
		for(size_t j=0; j<cols; j++){
			(*this)(i,j) = f((*this)(i,j));
		}
	}
}

// factories
template<Numeric T>
Matrix<T> Matrix<T>::zeros(size_t rows, size_t cols) {
    return Matrix<T>(rows, cols); // default constructor already zero-initializes
}

template<Numeric T>
Matrix<T> Matrix<T>::random(size_t rows, size_t cols, T low, T high) {
    Matrix<T> res(rows, cols);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<T> dist(low, high);
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            res(i, j) = dist(gen);
        }
    }
    return res;
}

template<Numeric T>
Matrix<T> Matrix<T>::he(size_t rows, size_t cols) {
    Matrix<T> res(rows, cols);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<T> dist(0, std::sqrt(static_cast<T>(2) / static_cast<T>(cols)));
    for (size_t i = 0; i < rows; i++)
        for (size_t j = 0; j < cols; j++)
            res(i, j) = dist(gen);
    return res;
}
