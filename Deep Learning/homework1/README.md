---
layout: default
title: Homework 1
permalink: /homework/01/index.html
---

## Homework 1

In this homework, we will practice **how to code in PyTorch**.
The homework is split into three parts:

- `pytorch_basics.py`
- `nearest_neighbor_classifier.py`
- `weather_forecast.py`

Before starting, feel free to play with this [tensor notebook](notebook.ipynb).
It should give you some examples on how to create and manipulate pytorch tensors.

*This homework must be done individually. Please refer to the honor code for more details.*

This homework should easily run on your local laptop / desktop without any GPU acceleration.

## Part 1: PyTorch Basics (40 points + 2 extra credit points)

Let's get familiar with how to use various PyTorch operations.
You will be given a number of python functions, each takes one or more `torch.Tensor`'s as an input and outputs a `torch.Tensor`.
However, these functions are not very efficient.
Your job is to re-implement the same functions using PyTorch.
In `pytorch_basics.py` file, there are 12+2 questions.
We provide *the number of characters* in our solution to give you an idea on how complex these requests are.
The character count includes the `return` statement, but no indentation whitespace.
All our solutions are one-liners.
Your solution can have any number of lines as long as you restrict yourself to only PyTorch functions.

Please refer to [official PyTorch documents](https://pytorch.org/docs/stable/torch.html) to learn more about various functionalities for tensor manipulation.

Relevant functions:

- [torch.abs](https://pytorch.org/docs/stable/generated/torch.abs.html)
- [torch.any](https://pytorch.org/docs/stable/generated/torch.any.html)
- [torch.cat](https://pytorch.org/docs/stable/generated/torch.cat.html)
- [torch.cumsum](https://pytorch.org/docs/stable/generated/torch.cumsum.html)
- [torch.diagonal](https://pytorch.org/docs/stable/generated/torch.diagonal.html)
- [torch.diff](https://pytorch.org/docs/stable/generated/torch.diff.html)
- [torch.flip](https://pytorch.org/docs/stable/generated/torch.flip.html)
- [torch.max](https://pytorch.org/docs/stable/generated/torch.max.html)
- [torch.mean](https://pytorch.org/docs/stable/generated/torch.mean.html)
- [torch.nonzero](https://pytorch.org/docs/stable/generated/torch.nonzero.html)
- [torch.stack](https://pytorch.org/docs/stable/generated/torch.stack.html)
- [torch.sum](https://pytorch.org/docs/stable/generated/torch.sum.html)
- [torch.unique](https://pytorch.org/docs/stable/generated/torch.unique.html)
- [torch.where](https://pytorch.org/docs/stable/generated/torch.where.html)
- [torch.zeros_like](https://pytorch.org/docs/stable/generated/torch.zeros_like.html)

Here are some **general rules** for this part:

- No loops (for, while, recursion)
- Only torch function calls
- No numpy; PyTorch and tensor operations only
- No assignments to results (e.g., `x[1] = 5; return x`)

Hint:

- Each solution requires less than 10 PyTorch commands

The grader will convert your code to TorchScript (a strict subset of python code that only supports basic PyTorch commands) to make sure your solutions conforms to the rules above.

## Part 2: Nearest Neighbor Classifier (30 points)

In **Part 2** of the homework, you will implement a basic machine learning algorithm in PyTorch: a *nearest neighbor classifier*.
In `nearest_neighbor_classifier.py`, we provide a starter code with descriptions of what you need to implement.
You will build a codebase to create, pre-process dataset, and find nearest neighbors of a given data.
**You are encouraged to use the PyTorch functions you converted in the Part 1.**
Keep in mind, your code should be *efficient*, and the solution is often very compact!

Relevant functions:

- All above
- [torch.as_tensor](https://pytorch.org/docs/stable/generated/torch.as_tensor.html)
- [torch.argmin](https://pytorch.org/docs/stable/generated/torch.argmin.html)
- [torch.pow](https://pytorch.org/docs/stable/generated/torch.pow.html)
- [torch.topk](https://pytorch.org/docs/stable/generated/torch.topk.html)


## **Part 3: Weather Forecasting (30 points)**

In **Part 3** of the homework, you will implement a simple predictive model for weather (temperature).
In `weather_forecast.py`, we provide a starter code with descriptions of what you need to implement.
You will build a simple interface to work with **days of weather measurement of Austin, Texas.**

Again, you are encouraged to use the PyTorch functions you converted in the Part 1.
Keep in mind, your code should be *efficient*, and the solution is often very compact!

Relevant functions:

- All above
- [torch.arange](https://pytorch.org/docs/stable/generated/torch.arange.html)
- [torch.Tensor.long](https://pytorch.org/docs/stable/generated/torch.Tensor.long.html)
- [torch.Tensor.float](https://pytorch.org/docs/stable/generated/torch.Tensor.float.html)


## Grading

The test grader we provide
```bash
python3 -m grader homework -v
```
will run a subset of test cases we use during the actual testing.
The point distributions will be the same, but we will use additional test cases.
The performance on the test grader may vary.

## Submission

Once you finished the assignment, create a submission bundle using
```bash
python3 bundle.py homework [YOUR UT ID]
```
and submit the zip file on canvas. Please note that the maximum file size our grader accepts is **20MB**. Please keep your model compact.
Please double-check that your zip file was properly created, by grading it again
```bash
python3 -m grader [YOUR UT ID].zip
```

## Online grader

We will use an automated grader through canvas to grade all your submissions. There is a soft limit of **5** submisisons per assignment. Please contact the course staff before going over this limit, otherwise your submission might be counted as invalid.

The online grading system will use a slightly modified version of python and the grader:

 * Please do not use the `exit` or `sys.exit` command, it will likely lead to a crash in the grader
 * Please do not try to access, read, or write files outside the ones specified in the assignment. This again will lead to a crash. File writing is disabled.
 * Network access is disabled. Please do not try to communicate with the outside world.
 * Forking is not allowed!
 * `print` or `sys.stdout.write` statements from your code are ignored and not returned.

Please do not try to break or hack the grader. Doing so will have negative consequences for your standing in this class and the program.

## Installation

We encourage using [Miniconda](https://docs.conda.io/en/latest/miniconda.html) to install the required packages.

```bash
conda create --name deeplearning python=3.11 -y
conda activate deeplearning
```

First, install [PyTorch](https://pytorch.org/get-started/locally/)

Then install additional dependencies:

```bash
pip install -r requirements.txt
```
