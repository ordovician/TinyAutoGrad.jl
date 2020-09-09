# Tiny Auto Gradient Library

This is a tiny library for computing the derivative of a function or the gradient if the function has multiple arguments. Oversimplified TinyAutoGrad does the same as [Zygote.jl](https://github.com/FluxML/Zygote.jl) and several other Julia libraries dealing with Automatic Differentiation. The key difference is that TinyAutoGrad is not intended for serious work. It is mainly intented to be educational.

The implementation is intentionally simple and limited so it is easier for a beginner to understand how Automatic Differentiation can be done with
dual numbers.

You can see one superficial explanation of this library in [this Nextjournal notebook](https://nextjournal.com/erik-engheim/implementation-of-automatic-differentiation).

## What is Automatic Differentiation?
It is a bit too big topic to cover here in a README, but it is also useful to give a vague idea so you can at least determine whether this is a topic you should be interested in our not.

For starters nearly all modern machine learning libraries such as Keras/TensorFlow, PyTorch and Flux use some kind of Automatic Differentiation under the hood to work their magic. Basically it is one of three common methods of caculating  the derivative of a function. You may remember this from math class. Here are some simple examples.

    f(x) = x²  then  f'(x) = 2x
    g(x) = x³  then  g'(x) = 3x² 
    h(x) = 4x  then  h'(x) = 4
    
This is what you probably did in high school and is what is called symbolic differentiation. You use some rules mathematicians have figured out to calculate the derivative of a function.

You where probably exposed to [numerical differentiation](https://en.wikipedia.org/wiki/Numerical_differentiation) in high school, which is an iterative approach to calculate the derivative of a function we don't have any rules to differentiate symbolically.

Then finally we have automatic differentiation, which is kind of the third method, which I was certainly not exposed to until learning about it in relation to machine learning. You may think about it as something between the numerical and symbolic approach.

The idea is that we use `Dual` numbers to keep track of what happens to the derivative as we perform various calculations.

## What is a Dual Number
A `Dual` number contains two parts, an `x` and an `ϵ`. The `x` is just a regular number. The `ϵ` however keeps track of what happens when you derivate.
