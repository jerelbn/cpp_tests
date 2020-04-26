// Create class to manipulate existing data without additional memory allocations
#include <iostream>
#include <math.h>

using namespace std;

class Vector
{
public:
    double* ptr = NULL;
    bool* internal_memory = NULL;

    Vector() {}

    ~Vector()
    {
        if (internal_memory) delete[] ptr;
    }

    Vector(double* const _ptr)
    {
        ptr = _ptr;
    }

    Vector(double w, double x, double y, double z)
    {
        internal_memory = new bool;
        ptr = new double[4];
        ptr[0] = w;
        ptr[1] = x;
        ptr[2] = y;
        ptr[3] = z;
    }

    void normalize()
    {
        double mag = sqrt(w()*w() + x()*x() + y()*y() + z()*z());
        ptr[0] /= mag;
        ptr[1] /= mag;
        ptr[2] /= mag;
        ptr[3] /= mag;
    }

    double& w() { return ptr[0]; }
    double& x() { return ptr[1]; }
    double& y() { return ptr[2]; }
    double& z() { return ptr[3]; }
};

int main()
{
    double a[4] = {1, 2, 3, 4};
    cout << "a_original = ";
    for (int i = 0; i < 4; ++i)
        cout << a[i] << " ";
    cout << endl;

    Vector b(a[0], a[1], a[2], a[3]);
    b.normalize();
    cout << "a_first = ";
    for (int i = 0; i < 4; ++i)
        cout << a[i] << " ";
    cout << endl;

    Vector c(a);
    c.normalize();
    cout << "a_second = ";
    for (int i = 0; i < 4; ++i)
        cout << a[i] << " ";
    cout << endl;
}