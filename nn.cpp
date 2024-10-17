#include <iostream>
#include <set>
#include <string>
#include <cmath>
#include <memory> // for shared pointers

struct value : public std::enable_shared_from_this<value>
{
    double data;
    double grad;
    std::multiset<std::shared_ptr<value>> prev;
    std::string op;
    std::string label;

    value() : data(0), grad(0) {}
    value(std::string lab) : data(0), grad(0), label(lab) {}
    value(double x) : data(x), grad(0) {}
    value(double x, std::string lab) : data(x), grad(0), label(lab) {}

    // back propogate gradient
    void backward()
    {
        if (this->op == "+")
        {
            for (const auto& child : this->prev)
            {
                child->grad += this->grad;  // Accumulate the gradient for addition
                child->backward();  // Recursive backward call
            }
        }
        else if (this->op == "*")
        {
            // Assuming 'this->prev' contains exactly two children (operands)
            auto child1 = *(this->prev.begin());       // Get first operand
            auto child2 = *(++this->prev.begin());     // Get second operand

            // Propagate gradients based on the multiplication
            child1->grad += this->grad * child2->data; // Gradient for the first operand
            child2->grad += this->grad * child1->data; // Gradient for the second operand

            // Recursive backward call for both children
            child1->backward();  
            child2->backward();
        }
        else if (this->op == "tanh")
        {
            for (const auto& child : this->prev)
            {
                child->grad += this->grad * (1 - std::pow(std::tanh(child->data), 2));  // Modify the original child
                child->backward();  // Recursive backward call
            }
        }
        else if (this->op == "exp")
        {
            for (const auto& child : this->prev)
            {
                std::cout << child->data << " " << this->grad << " " << (this->grad * (this->data)) << std::endl;
                child->grad += this->grad * std::exp(child->data);  // Modify the original child
                std::cout << child->grad << std::endl;
                child->backward();  // Recursive backward call
            }
        }
        else if (this->op == "pow")
        {
            // Assuming 'this->prev' contains the base and exponent
            auto base = *(this->prev.begin());
            auto exponent = *(++this->prev.begin());

            // Correct gradient calculations for the power operation
            double base_data = base->data;
            double exponent_data = exponent->data;

            // Gradient w.r.t. base: d(base^exponent)/d(base) = exponent * base^(exponent-1)
            base->grad += this->grad * exponent_data * std::pow(base_data, exponent_data - 1); 
            // Gradient w.r.t. exponent: d(base^exponent)/d(exponent) = base^exponent * log(base)
            if (base_data > 0) // Avoid log(0) and undefined behavior
            {
                exponent->grad += this->grad * std::pow(base_data, exponent_data) * std::log(base_data);
            }
            
            // Recursive backward call for both base and exponent
            base->backward();
            exponent->backward();
        }
    }

    // Comparison operator for std::set (ordering based on 'data' value)
    bool operator<(const value& other) const  // Change to reference
    {
        return data < other.data;  // Compare based on 'data' field
    }

    // Addition operator
    std::shared_ptr<value> operator+(double scalar)
    {
        auto temp = std::make_shared<value>(scalar, "scalar");
        auto result = *this + temp;
        return result;
    }
    std::shared_ptr<value> operator+(const std::shared_ptr<value>& other)
    {
        auto result = std::make_shared<value>(other->data + this->data);
        // Add the current object and 'other' to result's prev
        result->prev.insert(shared_from_this());  // Insert the current object
        result->prev.insert(other);  // Insert 'other'
        result->op = "+";

        return result; // Return shared pointer
    }

    // Subtraction operator
    std::shared_ptr<value> operator-(double scalar)
    {
        auto temp = std::make_shared<value>(scalar, "scalar");
        auto result = *this - temp;
        return result;
    }
    std::shared_ptr<value> operator-(const std::shared_ptr<value>& other)
    {
        return *this + (-1*other->data);
    }

    // Multiplication operator
    std::shared_ptr<value> operator*(double scalar)
    {
        auto temp = std::make_shared<value>(scalar, "scalar");
        auto result = *this * temp;
        return result;
    }
    std::shared_ptr<value> operator*(const std::shared_ptr<value>& other)
    {
        auto result = std::make_shared<value>(other->data * this->data);
        // Add the current object and 'other' to result's prev
        result->prev.insert(shared_from_this());  // Insert the current object
        result->prev.insert(other);  // Insert 'other'
        result->op = "*";

        return result; // Return shared pointer
    }

    // Division operator
    std::shared_ptr<value> operator/(double scalar)
    {
        auto temp = std::make_shared<value>(scalar, "scalar");
        auto result = *this / temp;
        return result;
    }
    std::shared_ptr<value> operator/(const std::shared_ptr<value>& other)
    {
        return *this * other->pow(-1);
    }

    std::shared_ptr<value> pow(double scalar)
    {
        auto temp = std::make_shared<value>(scalar, "scalar");
        return this->pow(temp);
    }

    std::shared_ptr<value> pow(const std::shared_ptr<value>& other)
    {
        double t = std::pow(this->data, other->data);
        auto result = std::make_shared<value>(t);
        result->prev.insert(shared_from_this());  // Insert the current object
        result->prev.insert(other);
        result->op = "pow";

        return result; // Return shared pointer
    }

    std::shared_ptr<value> tanh()
    {
        double t = std::tanh(this->data);
        auto result = std::make_shared<value>(t);
        result->prev.insert(shared_from_this());  // Insert the current object
        result->op = "tanh";

        return result; // Return shared pointer
    }

    std::shared_ptr<value> exp()
    {
        double t = std::exp(this->data);
        auto result = std::make_shared<value>(t);
        result->prev.insert(shared_from_this());  // Insert the current object
        result->op = "exp";

        return result; // Return shared pointer
    }

};

// must be outside of struct or class
std::shared_ptr<value> operator+(double scalar, const std::shared_ptr<value>& other) //equivalent to __roperation__ in python
{
    auto temp = std::make_shared<value>(scalar, "scalar");
    auto result = *other + temp;
    return result;
}
std::shared_ptr<value> operator-(double scalar, const std::shared_ptr<value>& other) //equivalent to __roperation__ in python
{
    auto temp = std::make_shared<value>(scalar, "scalar");
    auto result = *other - temp;
    return result;
}
std::shared_ptr<value> operator*(double scalar, const std::shared_ptr<value>& other) //equivalent to __roperation__ in python
{
    auto temp = std::make_shared<value>(scalar, "scalar");
    auto result = *other * temp;
    return result;
}

void print_expr(const value& val)
{
    std::cout << val.label << "=" << "(" << val.data << ", " << val.grad << ")";
    if (!val.op.empty()) // check if leaf node
    {
        std::cout << " [" << val.op << "]";
    }
    std::cout << std::endl;
    for (const auto& child : val.prev)
    {
        print_expr(*child);
    }
}

int main()
{

    /*
        WHEN USING OPERATORS:
            Left needs to be derefrenced shared pointer and right needs to be shared pointer
            c++ does not allow for (this) to be shared pointer
    */

    // auto x1 = std::make_shared<value>(2.0, "x1");
    // auto y1 = std::make_shared<value>(4.0, "y1");
    // auto z1 = *y1 - x1;
    // z1->label = "z1";
    
    // z1->grad = 1;
    // z1->backward();

    // print_expr(*z1);

    // auto a = std::make_shared<value>(3, "a");
    // auto b = a->exp();
    // b->label = "b";
    // b->grad = 1;
    // b->backward();
    // print_expr(*b);

    // define leave nodes as shared pointers
    auto x1 = std::make_shared<value>(2.0, "x1");
    auto x2 = std::make_shared<value>(0.0, "x2");

    auto w1 = std::make_shared<value>(-3.0, "w1");
    auto w2 = std::make_shared<value>(1.0, "w2");

    auto b = std::make_shared<value>(6.8814, "b");

    // define intermediate nodes with operations
    auto x1w1 = *x1 * w1;
    x1w1->label = "x1w1";
    auto x2w2 = *x2 * w2;
    x2w2->label = "x2w2";
    auto x1w1x2w2 = *x1w1 + x2w2;
    x1w1x2w2->label = "x1w1 + x2w2";

    auto n = *x1w1x2w2 + b;
    n->label = "n";

    auto e = (2*n)->exp();
    auto o = (*(*e-1))/(*e+1);
    o->label = "o";

    o->grad = 1;
    o->backward();
    
    print_expr(*o);


    return 0;
}
