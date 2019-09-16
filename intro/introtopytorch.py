#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 16:49:43 2019
Pytorch intro
@author: nyamewaa
"""

import torch
x=torch.ones(2,2,requires_grad=True) 
print(x)
#%%
y=x+2
print(y)
print(y.grad_fn)
z=y*y*3
out=z.mean()
print(z,out)
#%%
a=torch.randn(2,2)
a=((a*3)/(a-1))
print(a.requires_grad)
a.requires_grad_(True)
print(a.requires_grad)
b=(a*a).sum()   
print(b.grad_fn)

#%% AUTOGRAD
import torch
x = torch.ones(2,2, requires_grad=True)
y=x+2
print(y)
print(y.grad_fn)
z=y*y*3 
out=z.mean()
print(z,out)

a=torch.randn(2,2)
a=((a*3)/(a-1))
print(a.requires_grad)
a.requires_grad_(True)
print(a.requires_grad)
b=(a*a).sum()
print(b.grad_fn)

#%% GRADIENTS WITH AUTOGRAD
out.backward()
print(x.grad)

#vector Jacobian products
x=torch.rand(3, requires_grad=True)
y=x*2
while y.data.norm()<1000:
    y=y*2
print(y)

v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
y.backward(v)

print(x.grad)

print(x.requires_grad)
print((x ** 2).requires_grad)

with torch.no_grad():
    print((x ** 2).requires_grad)

