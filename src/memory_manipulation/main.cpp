// Create class to manipulate existing data without additional memory allocations
#include <iostream>
#include <math.h>

using namespace std;

class Quaternion
{
public:
    enum
    {
        _w,
        _x,
        _y,
        _z,
        _size
    };

    double buf[_size];

    Quaternion()
    {
        buf[_w] = 1.0;
        buf[_x] = 0.0;
        buf[_y] = 0.0;
        buf[_z] = 0.0;
    }

    Quaternion(const double& w, const double& x, const double& y, const double& z)
    {
        buf[_w] = w;
        buf[_x] = x;
        buf[_y] = y;
        buf[_z] = z;
    }

    void normalize()
    {
        double mag = sqrt(w()*w() + x()*x() + y()*y() + z()*z());
        buf[_w] /= mag;
        buf[_x] /= mag;
        buf[_y] /= mag;
        buf[_z] /= mag;
    }

    const double& w() const { return buf[0]; }
    const double& x() const { return buf[1]; }
    const double& y() const { return buf[2]; }
    const double& z() const { return buf[3]; }
};

class QuaternionMap
{
public:
    enum
    {
        _w,
        _x,
        _y,
        _z,
        _size
    };

    double * buf = NULL;

    QuaternionMap(double * ptr)
    { 
        buf = ptr;
    }

    void normalize()
    {
        double mag = sqrt(w()*w() + x()*x() + y()*y() + z()*z());
        buf[_w] /= mag;
        buf[_x] /= mag;
        buf[_y] /= mag;
        buf[_z] /= mag;
    }

    const double& w() const { return buf[0]; }
    const double& x() const { return buf[1]; }
    const double& y() const { return buf[2]; }
    const double& z() const { return buf[3]; }
};

int main()
{
    double a[4] = {1, 2, 3, 4};
    cout << "a_original = ";
    for (int i = 0; i < 4; ++i)
        cout << a[i] << " ";
    cout << endl;

    Quaternion b(a[0], a[1], a[2], a[3]);
    b.normalize();
    cout << "a_first = ";
    for (int i = 0; i < 4; ++i)
        cout << a[i] << " ";
    cout << endl;

    QuaternionMap c(a);
    c.normalize();
    cout << "a_second = ";
    for (int i = 0; i < 4; ++i)
        cout << a[i] << " ";
    cout << endl;
}