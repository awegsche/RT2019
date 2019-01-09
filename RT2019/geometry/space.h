#pragma once

#include <cuda_runtime.h>

#include <string>
#include <cmath>
#include <algorithm>
#include <sstream>

template<typename T>
class Vector3D
{
public:
    T x, y, z;

    __host__ __device__ Vector3D(T _x = 0.0f, T _y = 0.0f, T _z = 0.0f)
        : x(_x), y(_y), z(_z) {}

    // properties

    __host__ __device__ T squared_length() const;
    __host__ __device__ T length() const;

    __host__ __device__ Vector3D hat() const {
		T inv_len = 1.0 / length();
        return Vector3D(x * inv_len, y * inv_len, z * inv_len);
    }

    // arithmetic operators

    __host__ __device__ Vector3D operator+(const Vector3D& v) const {
        return Vector3D(x + v.x, y + v.y, z + v.z);
    }

    __host__ __device__ Vector3D operator-(const Vector3D& v) const {
        return Vector3D(x - v.x, y - v.y, z - v.z);
    }

    __host__ __device__ Vector3D operator*(T a) {
        return Vector3D(x * a, y * a, z* a);
    }

    __host__ __device__ Vector3D operator*(const Vector3D& v) const {
        return Vector3D(x * v.x, y * v.y, z * v.z);
    }
    __host__ __device__ Vector3D& operator *=(const Vector3D& v) {
        x *= v.x;
        y *= v.y;
        z *= v.z;
        return *this;
    }

    __host__ __device__ T dot(const Vector3D& v) const {
        return x * v.x + y * v.y + z * v.z;
    }
    __host__ __device__ Vector3D operator^(const Vector3D& v) const {
		return Vector3D(
			y * v.z - z * v.y,
			z * v.x - x * v.z,
			x * v.y - y * v.x);
    }

};

// non-member operators
template<typename T>
Vector3D<T> pmul(const Vector3D<T>& a, const Vector3D<T>& b);

template<typename T>
std::string vtos(const Vector3D<T>& v);




template<typename T>
T Vector3D<T>::squared_length() const
{
    return x*x + y*y + z*z;
}

template<typename T>
T Vector3D<T>::length() const
{
    return sqrt(squared_length());
}

template<typename T>
Vector3D<T> pmul(const Vector3D<T> &a, const Vector3D<T> &b)
{
    return a.pmul(b);
}

template<typename T>
std::string vtos(const Vector3D<T> &v)
{
    return static_cast<std::stringstream&>
        (
            std::ostringstream() << "(" << v.x << ", " << v.y << ", " << v.z << ")"
        ).str();
}


template<typename T>
class Vector2D
{
public:
	T x, y;

    __host__ __device__ Vector2D(T _x = 0.0f, T _y = 0.0f)
        : x(_x), y(_y) {}

    // properties

	__host__ __device__ T squared_length() const {
		return x * x + y * y;
	}
	__host__ __device__ T length() const {
		return sqrt(squared_length());
	}


    // arithmetic operators

    __host__ __device__ Vector2D operator+(const Vector2D& v) const {
        return Vector2D(x + v.x, y + v.y);
    }

    __host__ __device__ Vector2D operator-(const Vector2D& v) const {
        return Vector2D(x - v.x, y - v.y);
    }

    __host__ __device__ Vector2D operator*(T a) {
        return Vector2D(x * a, y * a);
    }

    __host__ __device__ Vector2D operator*(const Vector2D& v) const {
        return Vector2D(x * v.x, y * v.y);
    }
    __host__ __device__ Vector2D& operator *=(const Vector2D& v) {
        x *= v.x;
        y *= v.y;
        return *this;
    }

    __host__ __device__ T dot(const Vector2D& v) const {
        return x * v.x + y * v.y;
    }

};
