#include <catch2/catch_test_macros.hpp>
#include "matrix.hpp"
#include <cmath>

// ── Construction ──────────────────────────────────────────────────────────────

TEST_CASE("Default constructor zero-initializes", "[matrix][construction]") {
    Matrix<float> m(3, 4);
    REQUIRE(m.rows == 3);
    REQUIRE(m.cols == 4);
    for (size_t i = 0; i < m.rows; i++)
        for (size_t j = 0; j < m.cols; j++)
            CHECK(m(i, j) == 0.0f);
}

TEST_CASE("Src constructor copies data correctly", "[matrix][construction]") {
    float src[] = {1, 2, 3, 4, 5, 6};
    Matrix<float> m(2, 3, src);
    REQUIRE(m(0, 0) == 1.0f);
    REQUIRE(m(0, 1) == 2.0f);
    REQUIRE(m(1, 2) == 6.0f);
}

// ── Rule of 5 ─────────────────────────────────────────────────────────────────

TEST_CASE("Copy constructor is deep", "[matrix][rule_of_5]") {
    float src[] = {1, 2, 3, 4};
    Matrix<float> a(2, 2, src);
    Matrix<float> b(a);

    b(0, 0) = 99.0f;
    CHECK(a(0, 0) == 1.0f);  // a is unchanged
    CHECK(b(0, 0) == 99.0f);
}

TEST_CASE("Move constructor leaves source empty", "[matrix][rule_of_5]") {
    Matrix<float> a(3, 3);
    a(0, 0) = 5.0f;

    Matrix<float> b(std::move(a));
    CHECK(b(0, 0) == 5.0f);
    CHECK(a.rows == 0);
    CHECK(a.cols == 0);
}

TEST_CASE("Copy assignment is deep", "[matrix][rule_of_5]") {
    float src[] = {1, 2, 3, 4};
    Matrix<float> a(2, 2, src);
    Matrix<float> b(2, 2);
    b = a;

    b(0, 0) = 99.0f;
    CHECK(a(0, 0) == 1.0f);
    CHECK(b(0, 0) == 99.0f);
}

TEST_CASE("Move assignment leaves source empty", "[matrix][rule_of_5]") {
    Matrix<float> a(2, 2);
    a(0, 0) = 7.0f;

    Matrix<float> b(2, 2);
    b = std::move(a);
    CHECK(b(0, 0) == 7.0f);
    CHECK(a.rows == 0);
    CHECK(a.cols == 0);
}

// ── Element access ────────────────────────────────────────────────────────────

TEST_CASE("Element read and write", "[matrix][access]") {
    Matrix<float> m(3, 3);
    m(1, 2) = 42.0f;
    REQUIRE(m(1, 2) == 42.0f);
}

// ── Arithmetic ────────────────────────────────────────────────────────────────

TEST_CASE("Matrix addition", "[matrix][arithmetic]") {
    float a_data[] = {1, 2, 3, 4};
    float b_data[] = {5, 6, 7, 8};
    Matrix<float> a(2, 2, a_data);
    Matrix<float> b(2, 2, b_data);
    Matrix<float> c = a + b;

    CHECK(c(0, 0) == 6.0f);
    CHECK(c(0, 1) == 8.0f);
    CHECK(c(1, 0) == 10.0f);
    CHECK(c(1, 1) == 12.0f);
}

TEST_CASE("Scalar multiplication", "[matrix][arithmetic]") {
    float src[] = {1, 2, 3, 4};
    Matrix<float> a(2, 2, src);
    Matrix<float> b = a * 2.0f;

    CHECK(b(0, 0) == 2.0f);
    CHECK(b(0, 1) == 4.0f);
    CHECK(b(1, 0) == 6.0f);
    CHECK(b(1, 1) == 8.0f);
}

TEST_CASE("Square matmul 3x3", "[matrix][matmul]") {
    float a_data[] = {1,2,3, 4,5,6, 7,8,9};
    float b_data[] = {9,8,7, 6,5,4, 3,2,1};
    Matrix<float> a(3, 3, a_data);
    Matrix<float> b(3, 3, b_data);
    Matrix<float> c = a.matmul(b);

    // row 0 of a dot cols of b
    CHECK(c(0, 0) == 30.0f);
    CHECK(c(0, 1) == 24.0f);
    CHECK(c(0, 2) == 18.0f);
    CHECK(c(1, 0) == 84.0f);
    CHECK(c(2, 2) == 90.0f);
}

TEST_CASE("Non-square matmul 2x3 * 3x4", "[matrix][matmul]") {
    // A is 2x3, B is 3x4, result should be 2x4
    float a_data[] = {1,2,3, 4,5,6};
    float b_data[] = {1,2,3,4, 5,6,7,8, 9,10,11,12};
    Matrix<float> a(2, 3, a_data);
    Matrix<float> b(3, 4, b_data);
    Matrix<float> c = a.matmul(b);

    REQUIRE(c.rows == 2);
    REQUIRE(c.cols == 4);
    CHECK(c(0, 0) == 38.0f);
    CHECK(c(0, 1) == 44.0f);
    CHECK(c(1, 0) == 83.0f);
    CHECK(c(1, 3) == 128.0f);
}

// ── Transpose ─────────────────────────────────────────────────────────────────

TEST_CASE("Transpose square matrix", "[matrix][transpose]") {
    float src[] = {1,2,3,4};
    Matrix<float> a(2, 2, src);
    Matrix<float> t = a.transpose();

    CHECK(t(0, 0) == a(0, 0));
    CHECK(t(0, 1) == a(1, 0));
    CHECK(t(1, 0) == a(0, 1));
}

TEST_CASE("Transpose non-square 2x3", "[matrix][transpose]") {
    float src[] = {1,2,3, 4,5,6};
    Matrix<float> a(2, 3, src);
    Matrix<float> t = a.transpose();

    REQUIRE(t.rows == 3);
    REQUIRE(t.cols == 2);
    for (size_t i = 0; i < a.rows; i++)
        for (size_t j = 0; j < a.cols; j++)
            CHECK(t(j, i) == a(i, j));
}

// ── Apply ─────────────────────────────────────────────────────────────────────

TEST_CASE("apply returns new matrix", "[matrix][apply]") {
    float src[] = {1, 4, 9, 16};
    Matrix<float> a(2, 2, src);
    Matrix<float> b = a.apply([](float x) { return std::sqrt(x); });

    CHECK(b(0, 0) == 1.0f);
    CHECK(b(0, 1) == 2.0f);
    CHECK(b(1, 0) == 3.0f);
    CHECK(b(1, 1) == 4.0f);
    CHECK(a(0, 0) == 1.0f);  // original unchanged
}

TEST_CASE("apply_inplace modifies in place", "[matrix][apply]") {
    float src[] = {-1, 2, -3, 4};
    Matrix<float> a(2, 2, src);
    a.apply_inplace([](float x) { return x < 0 ? 0.0f : x; });  // ReLU

    CHECK(a(0, 0) == 0.0f);
    CHECK(a(0, 1) == 2.0f);
    CHECK(a(1, 0) == 0.0f);
    CHECK(a(1, 1) == 4.0f);
}

// ── Factories ─────────────────────────────────────────────────────────────────

TEST_CASE("zeros factory", "[matrix][factories]") {
    Matrix<float> m = Matrix<float>::zeros(3, 3);
    for (size_t i = 0; i < m.rows; i++)
        for (size_t j = 0; j < m.cols; j++)
            CHECK(m(i, j) == 0.0f);
}

TEST_CASE("random factory in range", "[matrix][factories]") {
    Matrix<float> m = Matrix<float>::random(10, 10, 0.0f, 1.0f);
    for (size_t i = 0; i < m.rows; i++)
        for (size_t j = 0; j < m.cols; j++) {
            CHECK(m(i, j) >= 0.0f);
            CHECK(m(i, j) <= 1.0f);
        }
}
