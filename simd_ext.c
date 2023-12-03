#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

float FloatAbs(float x) { return x * (x < 0 ? -1 : 1); }

typedef struct {
  int N;
  float *data;
} Matrix;

void MatrixMalloc(Matrix *mat, int n) {
  mat->N = n;
  mat->data = malloc(sizeof(float) * mat->N * mat->N);
}

void MatrixPrint(Matrix mat) {
  for (int i = 0; i < mat.N; ++i) {
    for (int j = 0; j < mat.N; ++j) {
      printf("%f ", mat.data[i * mat.N + j]);
    }
    printf("\n");
  }
}

void MatrixFree(Matrix *mat) { free(mat->data); }

void MatrixSetCoords(Matrix *mat, int row, int col, float value) {
  mat->data[row * mat->N + col] = value;
}

Matrix GetI(int n) {
  Matrix result;
  MatrixMalloc(&result, n);
  for (int row = 0; row < result.N; ++row) {
    for (int col = 0; col < result.N; ++col) {
      MatrixSetCoords(&result, row, col, row == col);
    }
  }
  return result;
}

void MatrixSetIndex(Matrix *mat, int index, float value) {
  mat->data[index] = value;
}

float MatrixGetCoords(Matrix *mat, int row, int col) {
  return mat->data[row * mat->N + col];
}

void MatrixPlus(Matrix *first, Matrix *second, Matrix *result) {
  for (int i = 0; i < result->N * result->N; ++i) {
    MatrixSetIndex(result, i, first->data[i] + second->data[i]);
  }
}

void MatrixMinus(Matrix *first, Matrix *second, Matrix *result) {
  for (int i = 0; i < result->N * result->N; ++i) {
    MatrixSetIndex(result, i, first->data[i] - second->data[i]);
  }
}
void MatrixTransponse(Matrix *mat, Matrix *result) {
  for (int i = 0; i < mat->N; ++i) {
    for (int j = 0; j < mat->N; ++j) {
      MatrixSetCoords(result, i, j, MatrixGetCoords(mat, j, i));
    }
  }
}

void MatrixMult(Matrix *first, Matrix *second, Matrix *result) {
  for (int i = 0; i < result->N; ++i) {
    // Создаем указатель на начало строки
    float *c = result->data + i * result->N;
    for (int j = 0; j < result->N; j += 8)
      _mm256_storeu_ps(c + j + 0, _mm256_setzero_ps());
    for (int k = 0; k < result->N; ++k) {
      const float *b = second->data + k * second->N;
      __m256 a = _mm256_set1_ps(first->data[i * first->N + k]);
      for (int j = 0; j < result->N; j += 8) {
        // fmadd_ps - Умножение
        //
        _mm256_storeu_ps(c + j + 0,
                         _mm256_fmadd_ps(a, _mm256_loadu_ps(b + j + 0),
                                         _mm256_loadu_ps(c + j + 0)));
      }
    }
  }
}

void MatrixScalarMult(Matrix *mat, Matrix *res, float scalar) {
  for (int i = 0; i < mat->N * mat->N; ++i) {
    MatrixSetIndex(res, i, mat->data[i] * scalar);
  }
}

float MatrixMaxRow(Matrix *mat) {
  float result = 0;
  for (int j = 0; j < mat->N; ++j) {
    float value = 0;
    for (int i = 0; i < mat->N; ++i) {
      value += FloatAbs(MatrixGetCoords(mat, i, j));
    }
    if (value > result) {
      result = value;
    }
  }
  return result;
}

float MatrixMaxCol(Matrix *mat) {
  float result = 0;
  for (int i = 0; i < mat->N; ++i) {
    float value = 0;
    for (int j = 0; j < mat->N; ++j) {
      value += FloatAbs(MatrixGetCoords(mat, i, j));
    }
    if (value > result) {
      result = value;
    }
  }
  return result;
}

Matrix GetB(Matrix *mat) {
  Matrix result;
  MatrixMalloc(&result, mat->N);
  float scalar = 1 / MatrixMaxRow(mat) / MatrixMaxCol(mat);
  MatrixTransponse(mat, &result);
  MatrixScalarMult(&result, &result, scalar);
  return result;
}

Matrix GetR(Matrix *mat, Matrix *B, Matrix *one) {
  Matrix result;
  Matrix ba;
  MatrixMalloc(&result, mat->N);
  MatrixMalloc(&ba, mat->N);
  MatrixMult(B, mat, &ba);
  MatrixMinus(one, &ba, &result);
  MatrixFree(&ba);
  return result;
}

void MatrixReverse(Matrix *mat, int m, Matrix *res) {
  Matrix result = GetI(mat->N);
  Matrix B = GetB(mat);
  Matrix R = GetR(mat, &B, &result);
  Matrix data = GetR(mat, &B, &result);

  for (int i = 0; i < m; ++i) {
    MatrixPlus(&result, &data, &result);
    MatrixMult(&data, &R, &data);
  }

  MatrixMult(&result, &B, res);
}

int main() {
  int M;
  if (!scanf("%d", &M)) {
    return 1;
  }

  Matrix mat;
  int n;
  if (!scanf("%d", &n)) {
    return 1;
  }
  MatrixMalloc(&mat, n);
  float value;
  for (int i = 0; i < mat.N * mat.N; ++i) {
    scanf("%f", &value);
    MatrixSetIndex(&mat, i, value);
  }
  clock_t startTime = clock();

  Matrix res;
  MatrixMalloc(&res, n);
  MatrixReverse(&mat, M, &res);
  clock_t endTime = clock();
  printf("%f\n", (float)(endTime - startTime) / CLOCKS_PER_SEC);

  MatrixFree(&mat);

  return 0;
}
